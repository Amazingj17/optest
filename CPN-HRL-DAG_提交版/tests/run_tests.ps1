param(
    [switch]$Full,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '../src')
try {
    if ($Full) {
        python -m pytest tests @PytestArgs
    } else {
        python -m pytest tests/test_hybrid_demo.py tests/test_independent_hybrid.py tests/test_advanced_search.py @PytestArgs
    }
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode
