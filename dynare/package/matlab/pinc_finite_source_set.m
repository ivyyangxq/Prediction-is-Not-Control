function [result, raw_json] = pinc_finite_source_set(payload, python_executable)
% Experimental finite four-class retained-set reporter (working source).
% JSON payload uses the Python reporter keyword arguments, including explicit
% alarm_status, candidate_classes and same-program banks. Does not simulate data.
% Returns retained classes and involvement implications, never a posterior probability.
% Use a Python environment containing this source revision; old releases lack it.
% Prefer a JSON file input to preserve singleton/list shapes exactly.
    if nargin ~= 2 || ~(isstruct(payload) || ischar(payload)) || ~ischar(python_executable)
        error('pinc:Input', 'Supply a payload struct or JSON file path and Python executable path.');
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
        if ischar(payload)
            pinc_json = fileread(payload);
        else
            pinc_json = jsonencode(payload);
        end
        fprintf(pinc_fid, '%s\n', pinc_json);
        fclose(pinc_fid);
    catch pinc_error
        fclose(pinc_fid);
        rethrow(pinc_error);
    end
    pinc_cmd = [pinc_finite_source_set_quote(python_executable) ' -m pinc_toolbox.finite_source_set_cli --input ' ...
                pinc_finite_source_set_quote(pinc_input) ' --output ' pinc_finite_source_set_quote(pinc_output)];
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

function quoted = pinc_finite_source_set_quote(value)
    q = char(39);
    replacement = [q char(34) q char(34) q];
    quoted = [q strrep(char(value), q, replacement) q];
end
