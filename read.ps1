# Perron One-Command Local Paper Reader
param(
    [switch]$Compile,
    [switch]$Watch,
    [switch]$Server,
    [int]$Port = 8080
)

$argsList = @()
if ($Compile) { $argsList += "--compile" }
if ($Watch) { $argsList += "--watch" }
if ($Server) { $argsList += "--server"; $argsList += "--port"; $argsList += $Port }

python paper/read_paper.py @argsList
