function [result, raw_json] = pinc_plan(request, python_executable)
% Explicit MDP planning via the installed PINC Python value-iteration core.
% REQUEST is a struct or JSON file with mdp.rewards/transitions/mask/beta.
% It evaluates the supplied model; it does not estimate G or detect switches.
    if nargin ~= 2 || ~(isstruct(request) || ischar(request)) || ~ischar(python_executable)
        error('pinc:Input', 'Supply an MDP request struct or JSON path and Python executable path.');
    end
    if isempty(python_executable) || python_executable(1) ~= '/'
        error('pinc:Configuration', 'Use an absolute POSIX Python path.');
    end
    pinc_work = tempname;
    mkdir(pinc_work);
    pinc_cleanup = onCleanup(@() rmdir(pinc_work, 's'));
    pinc_input = fullfile(pinc_work, 'input.json');
    pinc_output = fullfile(pinc_work, 'output.json');
    pinc_fid = fopen(pinc_input, 'w');
    if pinc_fid < 0, error('pinc:IO', 'Cannot write the request.'); end
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
    pinc_cmd = [pinc_plan_quote(python_executable) ' -m pinc_toolbox.planning_cli --input ' ...
                pinc_plan_quote(pinc_input) ' --output ' pinc_plan_quote(pinc_output)];
    [pinc_status, pinc_message] = system(pinc_cmd);
    if pinc_status ~= 0
        error('pinc:PythonFailed', 'Python exited %d: %s', pinc_status, pinc_message);
    end
    if ~exist(pinc_output, 'file')
        error('pinc:MissingOutput', 'Python returned without a result.');
    end
    raw_json = fileread(pinc_output);
    result = jsondecode(raw_json);
end

function quoted = pinc_plan_quote(value)
    q = char(39);
    replacement = [q char(34) q char(34) q];
    quoted = [q strrep(char(value), q, replacement) q];
end
