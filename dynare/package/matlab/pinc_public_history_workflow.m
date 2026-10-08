function [result, raw_json] = pinc_public_history_workflow(request, python_executable, state_path)
% Interactive JSON adapter for the explicit public-history workflow.
% Python owns fitting, region construction, source evidence, and atomic state.
    if nargin ~= 3 || ~(isstruct(request) || ischar(request)) || ~ischar(python_executable) || ~ischar(state_path)
        error('pinc:Input', 'Supply a request struct or JSON file path, Python executable, and state path.');
    end
    if isempty(python_executable) || python_executable(1) ~= '/' || ...
            isempty(state_path) || state_path(1) ~= '/'
        error('pinc:Configuration', 'Use absolute POSIX paths for Python and state.');
    end
    work = tempname;
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's')); %#ok<NASGU>
    input_path = fullfile(work, 'input.json');
    output_path = fullfile(work, 'output.json');
    fid = fopen(input_path, 'w');
    if fid < 0, error('pinc:IO', 'Cannot write workflow request.'); end
    try
        if ischar(request)
            request_json = fileread(request);
        else
            request_json = jsonencode(request);
        end
        fprintf(fid, '%s\n', request_json);
        fclose(fid);
    catch err
        fclose(fid);
        rethrow(err);
    end

    action = 'inspect';
    node = '';
    while true
        [result, raw_json] = pinc_public_history_call(python_executable, input_path, output_path, ...
                                          state_path, action, node);
        if isstruct(result) && isfield(result, 'summary') && iscell(result.summary)
            for i = 1:numel(result.summary), fprintf('%s\n', result.summary{i}); end
        else
            disp(result);
        end
        if isfield(result, 'error')
            error('pinc:WorkflowFailed', '%s', result.error.message);
        end
        if strcmp(action, 'stop') || strcmp(action, 'more_data') || ...
                (isfield(result, 'snapshot') && strcmp(result.snapshot.status, 'resolved'))
            return;
        end
        choices = result.snapshot.choices;
        if ~isstruct(choices) || isempty(choices)
            error('pinc:WorkflowProtocol', 'Python returned no permitted choices.');
        end
        fprintf('Choose one action:\n');
        for i = 1:numel(choices), fprintf('  %d: %s\n', i, choices(i).label); end
        raw = strtrim(input('Choice number: ', 's'));
        index = str2double(raw);
        if ~isfinite(index) || index < 1 || index > numel(choices) || index ~= floor(index)
            fprintf(2, 'Unknown choice "%s".\n', raw);
            action = 'inspect'; node = ''; continue;
        end
        action = choices(index).action;
        node = '';
        if isfield(choices(index), 'node') && ~isempty(choices(index).node)
            node = choices(index).node;
        end
    end
end

function [result, raw_json] = pinc_public_history_call(python_executable, input_path, output_path, state_path, action, node)
    if exist(output_path, 'file'), delete(output_path); end
    command = [pinc_public_history_quote(python_executable) ...
        ' -m pinc_toolbox.public_history_workflow_cli --input ' ...
        pinc_public_history_quote(input_path) ' --state ' pinc_public_history_quote(state_path) ...
        ' --output ' pinc_public_history_quote(output_path) ' --action ' action];
    if ~isempty(node), command = [command ' --node ' pinc_public_history_quote(node)]; end
    [status, message] = system(command);
    if ~exist(output_path, 'file')
        error('pinc:MissingOutput', 'Python returned no workflow result: %s', message);
    end
    raw_json = fileread(output_path);
    result = jsondecode(raw_json);
    if status ~= 0 && ~isfield(result, 'error')
        error('pinc:PythonFailed', 'Python exited %d: %s', status, message);
    end
end

function quoted = pinc_public_history_quote(value)
    quote = char(39);
    replacement = [quote char(34) quote char(34) quote];
    quoted = [quote strrep(char(value), quote, replacement) quote];
end
