function receipt = pinc_ks_saved_exhibits(input_dir, output_dir, python_executable)
% Regenerate saved KS CSV exhibits through Python; no scientific host is run.
    if nargin ~= 3 || isempty(python_executable) || python_executable(1) ~= '/'
        error('pinc:Input', 'Supply saved CSV directory, NEW output directory, absolute installed Python.');
    end
    cmd = [pinc_shell_quote(python_executable) ' -I -m pinc_toolbox.ks_saved_exhibits_cli --input-dir ' ...
           pinc_shell_quote(input_dir) ' --output-dir ' pinc_shell_quote(output_dir)];
    [status, message] = system(cmd);
    if status ~= 0, error('pinc:PythonFailed', '%s', message); end
    receipt = jsondecode(fileread(fullfile(output_dir, 'RECEIPT.json')));
end
