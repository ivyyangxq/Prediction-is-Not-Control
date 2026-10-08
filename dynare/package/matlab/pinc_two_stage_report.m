function [result, raw_json, report_text] = pinc_two_stage_report(request, python_executable, html_output)
% Shared production two-stage Python API for MATLAB/Octave/Dynare hosts.
% request is a JSON file path or struct. Python owns all statistical decisions.
    if nargin < 2 || nargin > 3 || ~(isstruct(request) || ischar(request)) || ~ischar(python_executable)
        error('pinc:Input', 'Supply a JSON request file or struct and Python executable.');
    end
    if isempty(python_executable) || python_executable(1) ~= '/'
        error('pinc:Configuration', 'Use an absolute POSIX Python executable path.');
    end
    if isstruct(request) && isfield(request, 'context') && isstruct(request.context) ...
            && isfield(request.context, 'route') && strcmp(request.context.route, 'conditional_history_saved_report')
        error('pinc:SavedTerminalJSONFile', ...
            'For conditional_history_saved_report, supply the original saved JSON file path. Struct JSON conversion cannot preserve null distinctly from empty arrays.');
    end
    work = tempname;
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's')); %#ok<NASGU>
    input_path = fullfile(work, 'request.json');
    output_path = fullfile(work, 'report.json');
    text_path = fullfile(work, 'report.txt');
    if ischar(request), payload = fileread(request); else, payload = jsonencode(request); end
    fid = fopen(input_path, 'w');
    if fid < 0, error('pinc:IO', 'Cannot write temporary request.'); end
    fprintf(fid, '%s\n', payload);
    fclose(fid);
    command = [pinc_report_quote(python_executable) ...
        ' -m pinc_toolbox.two_stage_report_cli --input ' pinc_report_quote(input_path) ...
        ' --output ' pinc_report_quote(output_path) ' --text-output ' pinc_report_quote(text_path)];
    if nargin == 3 && ~isempty(html_output)
        if ~ischar(html_output) || html_output(1) ~= '/'
            error('pinc:Configuration', 'Use an absolute POSIX HTML output path.');
        end
        if ischar(request) && strcmp(request, html_output)
            error('pinc:Configuration', 'HTML output must differ from the request path.');
        end
        command = [command ' --html-output ' pinc_report_quote(html_output)];
    end
    [status, message] = system(command);
    if status ~= 0
        error('pinc:PythonFailed', 'Python exited %d: %s', status, message);
    end
    if ~exist(output_path, 'file') || ~exist(text_path, 'file')
        error('pinc:MissingOutput', 'Python returned without both reports.');
    end
    raw_json = fileread(output_path);
    result = jsondecode(raw_json);
    report_text = fileread(text_path);
end

function quoted = pinc_report_quote(value)
    quote = char(39);
    quoted = [quote strrep(value, quote, [quote '"' quote '"' quote]) quote];
end
