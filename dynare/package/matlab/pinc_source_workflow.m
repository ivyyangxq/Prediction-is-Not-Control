function [result, raw_json] = pinc_source_workflow(request, python_executable, state_path)
% Interactive POSIX adapter for the one-node-at-a-time Python source workflow.
% The adapter owns only the menu and temporary JSON transport; Python owns
% source evaluation, state identity, and atomic state persistence.
    if nargin ~= 3 || ~(isstruct(request) || ischar(request)) || ~ischar(python_executable) || ~ischar(state_path)
        error('pinc:Input', 'Supply a request struct or JSON file path, Python executable, and state path.');
    end
    if isempty(python_executable) || python_executable(1) ~= '/' || isempty(state_path) || state_path(1) ~= '/'
        error('pinc:Configuration', 'Use absolute POSIX paths for Python and state.');
    end
    pinc_work = tempname;
    mkdir(pinc_work);
    pinc_cleanup = onCleanup(@() rmdir(pinc_work, 's')); %#ok<NASGU>
    pinc_input = fullfile(pinc_work, 'input.json');
    pinc_output = fullfile(pinc_work, 'output.json');
    pinc_fid = fopen(pinc_input, 'w');
    if pinc_fid < 0, error('pinc:IO', 'Cannot write the workflow request.'); end
    try
        if ischar(request)
            pinc_json = fileread(request);
        else
            pinc_json = jsonencode(request);
        end
        fprintf(pinc_fid, '%s\n', pinc_json);
        fclose(pinc_fid);
    catch pinc_error
        fclose(pinc_fid);
        rethrow(pinc_error);
    end

    action = 'inspect';
    node = '';
    while true
        [result, raw_json] = pinc_source_workflow_call(python_executable, pinc_input, pinc_output, ...
                                           state_path, action, node);
        if isstruct(result) && isfield(result, 'summary') && iscell(result.summary)
            for summary_index = 1:numel(result.summary)
                fprintf('%s\n', result.summary{summary_index});
            end
        else
            disp(result);
        end
        if strcmp(action, 'stop') || strcmp(action, 'more_data') || ...
                (isstruct(result) && isfield(result, 'snapshot') && ...
                 isfield(result.snapshot, 'status') && strcmp(result.snapshot.status, 'resolved'))
            return;
        end
        if isstruct(result) && isfield(result, 'error')
            error('pinc:WorkflowFailed', '%s', result.error.message);
        end
        if ~isstruct(result) || ~isfield(result, 'snapshot') || ~isfield(result.snapshot, 'choices')
            error('pinc:WorkflowProtocol', 'Python result has no permitted choice list.');
        end
        choices = result.snapshot.choices;
        if ~isstruct(choices) || isempty(choices)
            error('pinc:WorkflowProtocol', 'Python returned an empty permitted choice list.');
        end
        fprintf('Choose one action:\n');
        for choice_index = 1:numel(choices)
            fprintf('  %d: %s\n', choice_index, choices(choice_index).label);
        end
        choice_text = strtrim(input('Choice number: ', 's'));
        choice_index = str2double(choice_text);
        if ~isfinite(choice_index) || choice_index < 1 || choice_index > numel(choices) || choice_index ~= floor(choice_index)
            fprintf(2, 'Unknown choice "%s". Enter one of the displayed numbers.\n', choice_text);
            action = 'inspect';
            node = '';
            continue;
        end
        action = choices(choice_index).action;
        node = '';
        if isfield(choices(choice_index), 'node') && ~isempty(choices(choice_index).node)
            node = choices(choice_index).node;
        end
    end
end

function [result, raw_json] = pinc_source_workflow_call(python_executable, input_path, output_path, state_path, action, node)
    if exist(output_path, 'file')
        delete(output_path);
    end
    command = [pinc_source_workflow_quote(python_executable) ' -m pinc_toolbox.source_workflow_cli --input ' ...
               pinc_source_workflow_quote(input_path) ' --state ' pinc_source_workflow_quote(state_path) ...
               ' --output ' pinc_source_workflow_quote(output_path) ' --action ' action];
    if ~isempty(node)
        command = [command ' --node ' pinc_source_workflow_quote(node)];
    end
    [status, message] = system(command);
    if status ~= 0 && ~exist(output_path, 'file')
        error('pinc:PythonFailed', 'Python exited %d: %s', status, message);
    end
    if ~exist(output_path, 'file')
        error('pinc:MissingOutput', 'Python returned without a workflow result.');
    end
    raw_json = fileread(output_path);
    result = jsondecode(raw_json);
    if status ~= 0
        if isstruct(result) && isfield(result, 'error')
            error('pinc:PythonFailed', '%s', result.error.message);
        end
        error('pinc:PythonFailed', 'Python exited %d without a structured error: %s', status, message);
    end
end

function quoted = pinc_source_workflow_quote(value)
    q = char(39);
    replacement = [q char(34) q char(34) q];
    quoted = [q strrep(char(value), q, replacement) q];
end
