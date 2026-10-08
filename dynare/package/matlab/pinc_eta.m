function [result, raw_json] = pinc_eta(payload, python_executable)
% Shared Python eta/reference/source core; no .mod parser or duplicate math.
% POSIX MATLAB/Octave wrapper. JSON file input preserves singleton arrays.
% Second output preserves the original response JSON for lossless persistence.
% PAYLOAD explicitly supplies observed transitions, release clocks, declared
% feature dictionary, tasks, block-noise assumptions and source family.
% python_executable must point to an environment containing the NEW package.
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
    pinc_cmd = [pinc_eta_quote(python_executable) ' -m pinc_toolbox.public_eta_cli --input ' ...
                pinc_eta_quote(pinc_input) ' --output ' pinc_eta_quote(pinc_output)];
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

function quoted = pinc_eta_quote(value)
    q = char(39);
    replacement = [q char(34) q char(34) q];
    quoted = [q strrep(char(value), q, replacement) q];
end
