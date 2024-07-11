classdef ESO
    properties
        cache
        ionized_areas_index
        ionized_areas_positions
        storm_power
        bounds
        dim
        function_handle
        function_evaluations
        function_evaluations_list
        lightning
        n_lightning
        objective
        iterations
        iteration_current
        field_resistance
        field_intensity
        ke
        verbose
        stagnation
        objective_values
        iteration_times
        best_positions
        history
        max_evaluations
        field_intensity_history
        field_resistance_history
        field_elasticity_history
        storm_power_history
    end
    
    methods
        function obj = ESO(function_handle, pop_size, max_iter, max_eval, objective, bounds, verbose)
            if nargin < 7, verbose = true; end
            if nargin < 6, bounds = [0, 1]; end
            if nargin < 5, objective = 'min'; end
            if nargin < 4, max_eval = 500000; end
            if nargin < 3, max_iter = 1000; end
            if nargin < 2, pop_size = 50; end
            
            obj.cache = containers.Map();
            obj.ionized_areas_index = [];
            obj.ionized_areas_positions = [];
            obj.storm_power = 0;
            obj.bounds = bounds;
            obj.dim = size(bounds, 1);
            obj.function_handle = function_handle;
            obj.function_evaluations = 0;
            obj.function_evaluations_list = [];
            obj.lightning = arrayfun(@(~) obj.initialize_lightning(), 1:pop_size, 'UniformOutput', false);
            obj.n_lightning = pop_size;
            obj.objective = objective;
            obj.iterations = max_iter;
            obj.iteration_current = 0;
            obj.field_resistance = 0;
            obj.field_intensity = 0;
            obj.ke = 0;
            obj.verbose = verbose;
            obj.stagnation = zeros(pop_size, 1);
            obj.objective_values = [];
            obj.iteration_times = [];
            obj.best_positions = [];
            obj.history = cell(pop_size, 1);
            obj.max_evaluations = max_eval;
            obj.field_intensity_history = [];
            obj.field_resistance_history = [];
            obj.field_elasticity_history = [];
            obj.storm_power_history = [];
        end
        
        function value = exp(obj, x)
            if x > 10e1
                value = inf;
            elseif x < -10e1
                value = 0.0;
            else
                value = exp(x);
            end
        end
        
        function value = log(obj, x)
            value = log(x);
        end
        
        function obj = identify_ionized_areas(obj)
            [~, sorted_indices] = sort(cellfun(@(light) light.best_score, obj.lightning), 'descend');
            if strcmp(obj.objective, 'min')
                sorted_indices = flip(sorted_indices);
            end
            num_ionized = ceil(obj.n_lightning * (obj.field_resistance / 2));
            obj.ionized_areas_index = sorted_indices(1:num_ionized);
            obj.ionized_areas_positions = cellfun(@(idx) obj.lightning{idx}.position, num2cell(obj.ionized_areas_index), 'UniformOutput', false);
        end
        
        function lightning = initialize_lightning(obj)
            if ~isempty(obj.ionized_areas_positions)
                idx = randi(length(obj.ionized_areas_positions));
                selected_area = obj.ionized_areas_positions{idx};
                position = selected_area + obj.storm_power;
                position = min(max(position, obj.bounds(:, 1)'), obj.bounds(:, 2)');
            else
                position = (obj.bounds(:, 2)' - obj.bounds(:, 1)') .* rand(1, obj.dim) + obj.bounds(:, 1)';
            end
            
            pos_key = mat2str(position);
            if isKey(obj.cache, pos_key)
                best_score = obj.cache(pos_key);
            else
                best_score = obj.function_handle(position);
                obj.cache(pos_key) = best_score;
                obj.function_evaluations = obj.function_evaluations + 1;
                obj.function_evaluations_list = [obj.function_evaluations_list, obj.function_evaluations];
            end
            
            lightning.position = position;
            lightning.best_position = position;
            lightning.best_score = best_score;
        end
        
        function obj = calculate_field_resistance(obj)
            positions = cell2mat(cellfun(@(ray) ray.position, obj.lightning, 'UniformOutput', false)');
            max_diff = max(range(positions), 1e-50);
            obj.field_resistance = std(positions) / max_diff;
        end
        
        function obj = calculate_ke(obj)
            beta = 1 / (1 + obj.exp(-(obj.exp(obj.field_resistance) / (obj.field_resistance + 10e-50)) * ...
                (obj.field_resistance - abs(obj.log(1 - obj.field_resistance)))));
            obj.ke = obj.exp(obj.field_resistance) + (obj.exp(1 - obj.field_resistance) * abs(obj.log(obj.field_resistance + 10e-50))) * beta;
        end
        
        function obj = adjust_field_intensity(obj)
            iteration_factor = obj.iteration_current / max(obj.iterations, 1);
            gamma = 1 / (1 + obj.exp(-(obj.exp(obj.field_resistance) / (obj.field_resistance + 10e-50)) * ...
                (obj.field_resistance - abs(obj.log(1 - iteration_factor)))));
            obj.field_intensity = 10e-50 + obj.ke * gamma;
        end
        
        function obj = calculate_storm_power(obj)
            obj.storm_power = obj.field_resistance * obj.field_intensity ^ obj.ke;
        end
        
        function new_position = branch_and_propagate(obj, idx)
            if obj.stagnation(idx) > round(obj.ke)
                obj.stagnation(idx) = 0;
                new_position = obj.initialize_lightning().position;
                return;
            end
            
            if any(idx == obj.ionized_areas_index)
                new_position = obj.lightning{idx}.position * obj.storm_power;
                new_position = min(max(new_position, obj.bounds(:, 1)'), obj.bounds(:, 2)');
            else
                if ~isempty(obj.ionized_areas_positions)
                    new_position = zeros(1, obj.dim);
                    for i = 1:length(obj.ionized_areas_positions)
                        perturbation = (2 * rand(1, obj.dim) - 1) * obj.ke;
                        new_position = new_position + (obj.ionized_areas_positions{i} + (perturbation * obj.storm_power * obj.exp(obj.ke)));
                    end
                    new_position = new_position / length(obj.ionized_areas_positions);
                else
                    new_position = obj.initialize_lightning().position;
                end
                new_position = min(max(new_position, obj.bounds(:, 1)'), obj.bounds(:, 2)');
            end
        end
        
        function obj = select_path(obj, idx, adapted_path)
            pos_key = mat2str(adapted_path);
            if isKey(obj.cache, pos_key)
                adapted_score = obj.cache(pos_key);
            else
                adapted_score = obj.function_handle(adapted_path);
                obj.cache(pos_key) = adapted_score;
                obj.function_evaluations = obj.function_evaluations + 1;
                obj.function_evaluations_list = [obj.function_evaluations_list, obj.function_evaluations];
            end
            
            if (strcmp(obj.objective, 'min') && adapted_score < obj.lightning{idx}.best_score) || ...
               (strcmp(obj.objective, 'max') && adapted_score > obj.lightning{idx}.best_score)
                obj.lightning{idx}.position = adapted_path;
                obj.lightning{idx}.best_position = adapted_path;
                obj.lightning{idx}.best_score = adapted_score;
                obj.stagnation(idx) = 0;
            else
                obj.stagnation(idx) = obj.stagnation(idx) + 1;
            end
        end
        
        function [global_best_position, global_best_score] = update_optimal_strike(obj, idx, global_best_position, global_best_score)
            if (strcmp(obj.objective, 'min') && obj.lightning{idx}.best_score < global_best_score) || ...
               (strcmp(obj.objective, 'max') && obj.lightning{idx}.best_score > global_best_score)
                global_best_position = obj.lightning{idx}.best_position;
                global_best_score = obj.lightning{idx}.best_score;
            end
        end
        
        function [global_best_position, global_best_score] = optimize(obj)
            if ~isa(obj.function_handle, 'function_handle')
                error('Function must be callable.');
            end
            
            global_best_position = obj.lightning{1}.position;
            global_best_score = obj.lightning{1}.best_score;
            
            for iteration = 1:obj.iterations
                if obj.function_evaluations >= obj.max_evaluations
                    disp('Max evaluations reached, stopping optimization.');
                    break;
                end
                
                obj.iteration_current = iteration;
                iteration_start_time = tic;
                obj = obj.identify_ionized_areas();
                obj = obj.adjust_field_intensity();
                obj = obj.calculate_field_resistance();
                obj = obj.calculate_ke();
                obj = obj.calculate_storm_power();
                
                for idx = 1:obj.n_lightning
                    adapted_path = obj.branch_and_propagate(idx);
                    obj = obj.select_path(idx, adapted_path);
                    [global_best_position, global_best_score] = obj.update_optimal_strike(idx, global_best_position, global_best_score);
                end
                
                iteration_duration = toc(iteration_start_time);
                obj.iteration_times = [obj.iteration_times, iteration_duration];
                obj.objective_values = [obj.objective_values, global_best_score];
                obj.best_positions = [obj.best_positions; global_best_position];
                obj.field_intensity_history = [obj.field_intensity_history, obj.field_intensity];
                obj.field_resistance_history = [obj.field_resistance_history, obj.field_resistance];
                obj.field_elasticity_history = [obj.field_elasticity_history, obj.ke];
                obj.storm_power_history = [obj.storm_power_history, obj.storm_power];
                
                if obj.verbose
                    fprintf('Iteration: %d, Storm Power: %.5f, Field Intensity: %.3f, Field Resistance: %.3f, ke: %.3f, Current Best: %.20e, Global Best: %.20e, Iteration Time: %.5f seconds, Func. Eval: %d\n', ...
                        iteration, obj.storm_power, obj.field_intensity, obj.field_resistance, obj.ke, obj.lightning{1}.best_score, global_best_score, iteration_duration, obj.function_evaluations);
                end
            end
        end
    end
end
