# Perron One-Command Master Build & Publication Pipeline
param(
    [switch]$All,
    [switch]$Figures,
    [string]$Fig,
    [switch]$Compile,
    [switch]$Verify,
    [switch]$Test,
    [switch]$Read,
    [switch]$Watch,
    [switch]$Server,
    [int]$Port = 8080
)

$argsList = @()
if ($All) { $argsList += "--all" }
if ($Figures) { $argsList += "--figures" }
if ($Fig) { $argsList += "--fig"; $argsList += $Fig }
if ($Compile) { $argsList += "--compile" }
if ($Verify) { $argsList += "--verify" }
if ($Test) { $argsList += "--test" }
if ($Read) { $argsList += "--read" }
if ($Watch) { $argsList += "--watch" }
if ($Server) { $argsList += "--server"; $argsList += "--port"; $argsList += $Port }

if ($argsList.Count -eq 0) {
    $argsList += "--compile"
}

python scripts/build_paper.py @argsList
