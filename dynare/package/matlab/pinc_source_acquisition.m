function [result, raw_json] = pinc_source_acquisition(request, python_executable)
% Recommend an action from a qualified query snapshot; does not execute it.
    if ~ischar(python_executable) || isempty(python_executable) || python_executable(1) ~= '/'
        error('pinc:Configuration', 'Supply absolute Python executable path.');
    end
    work = tempname; mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's'));
    input_path = fullfile(work, 'input.json'); output_path = fullfile(work, 'output.json');
    if ischar(request), payload = fileread(request); else, payload = jsonencode(request); end
    fid = fopen(input_path, 'w'); fprintf(fid, '%s', payload); fclose(fid);
    command = [pinc_shell_quote(python_executable) ' -m pinc_toolbox.source_acquisition_cli --input ' ...
        pinc_shell_quote(input_path) ' --output ' pinc_shell_quote(output_path)];
    [status, message] = system(command);
    if status ~= 0, error('pinc:PythonFailed', '%s', message); end
    raw_json = fileread(output_path);
    result = jsondecode(raw_json);
end
