function quoted = pinc_shell_quote(value)
% POSIX argument quoting for the explicitly supported macOS/Octave example.
    q = char(39);
    replacement = [q char(34) q char(34) q];
    quoted = [q strrep(char(value), q, replacement) q];
end
