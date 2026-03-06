$ErrorActionPreference = "Stop"

# MiKTeX can fail if PATH contains malformed entries (example: paths ending in "paraview.exe\").
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and ($_ -notmatch 'paraview\.exe') }) -join ';'

Push-Location "$PSScriptRoot\\src"
try {
  xelatex -interaction=nonstopmode -halt-on-error lit_review.tex | Out-Null
  xelatex -interaction=nonstopmode -halt-on-error lit_review.tex | Out-Null
  xelatex -interaction=nonstopmode -halt-on-error body_only.tex | Out-Null
  xelatex -interaction=nonstopmode -halt-on-error body_only.tex | Out-Null
} finally {
  Pop-Location
}

Copy-Item -Force -LiteralPath "$PSScriptRoot\\src\\lit_review.pdf" -Destination "$PSScriptRoot\\lit_review.pdf"
Copy-Item -Force -LiteralPath "$PSScriptRoot\\src\\body_only.pdf" -Destination "$PSScriptRoot\\body_only.pdf"

Write-Host "Wrote:"
Write-Host "  $PSScriptRoot\\lit_review.pdf"
Write-Host "  $PSScriptRoot\\body_only.pdf"

