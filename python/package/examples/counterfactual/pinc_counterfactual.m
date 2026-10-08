function [report, raw_json] = pinc_counterfactual(request_path, host_file, output_path, python_executable)
% Run the installed full RND2 four-path Python kernel; host_file is trusted code.
% Empty host_file formats request.counterfactual_result only: no fit or host step.
% Optional request.pair_routing is evaluated by the same Python report function.
    if nargin ~= 4
        error('pinc:Input', 'Supply request JSON, explicit host Python file, output JSON and absolute Python.');
    end
    if isempty(python_executable) || python_executable(1) ~= '/'
        error('pinc:Configuration', 'Python executable must be an absolute POSIX path.');
    end
    if isempty(host_file)
        mode = ' --report-only';
    else
        mode = [' --host-file ' cf_quote(host_file)];
    end
    cmd = [cf_quote(python_executable) ' -I -m pinc_toolbox.counterfactual_cli --input ' ...
        cf_quote(request_path) mode ' --output ' cf_quote(output_path)];
    [status, message] = system(cmd);
    if status ~= 0, error('pinc:PythonFailed', '%s', message); end
    raw_json = fileread(output_path);
    report = jsondecode(raw_json);
end
function quoted = cf_quote(value)
    quote = char(39);
    quoted = [quote strrep(value, quote, [quote '"' quote '"' quote]) quote];
end
