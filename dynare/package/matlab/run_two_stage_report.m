% Bridge a request assembled from a Dynare measurement run into the shared
% Python API. Stage-1 and source blocks are supplied in the JSON request; this
% wrapper does not infer an alarm or create a statistical guarantee.
pinc_python = getenv('PINC_INSTALLED_PY');
pinc_input = getenv('PINC_TWO_STAGE_INPUT');
pinc_output = getenv('PINC_TWO_STAGE_OUTPUT');
if isempty(pinc_python) || isempty(pinc_input) || isempty(pinc_output)
    error('pinc:Configuration', ...
        'Set PINC_INSTALLED_PY, PINC_TWO_STAGE_INPUT and PINC_TWO_STAGE_OUTPUT.');
end
if pinc_python(1) ~= '/' || pinc_input(1) ~= '/' || pinc_output(1) ~= '/'
    error('pinc:Configuration', 'Use absolute POSIX paths.');
end
if ~exist(pinc_python, 'file') || ~exist(pinc_input, 'file')
    error('pinc:Configuration', 'Configured Python executable or request JSON is missing.');
end
pinc_example_dir = fileparts(mfilename('fullpath'));
addpath(pinc_example_dir);
pinc_text = [pinc_output '.txt'];
pinc_command = [pinc_shell_quote(pinc_python) ...
    ' -m pinc_toolbox.two_stage_report_cli --input ' pinc_shell_quote(pinc_input) ...
    ' --output ' pinc_shell_quote(pinc_output) ...
    ' --text-output ' pinc_shell_quote(pinc_text)];
pinc_html = getenv('PINC_TWO_STAGE_HTML');
if ~isempty(pinc_html)
    if pinc_html(1) ~= '/'
        error('pinc:Configuration', 'PINC_TWO_STAGE_HTML must be an absolute path.');
    end
    pinc_command = [pinc_command ' --html-output ' pinc_shell_quote(pinc_html)];
end
[pinc_exit, pinc_console] = system(pinc_command);
fprintf('%s\n', pinc_console);
if pinc_exit ~= 0
    error('pinc:PythonFailed', 'Two-stage report CLI exited %d.', pinc_exit);
end
if ~exist(pinc_output, 'file') || ~exist(pinc_text, 'file')
    error('pinc:PythonFailed', 'CLI exited without both requested report files.');
end
fprintf('PINC_TWO_STAGE_REPORT_COMPLETE output=%s text=%s\n', pinc_output, pinc_text);
