param([string]$InputDoc,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$wordApp=New-Object -ComObject Word.Application
$wordApp.Visible=$false
$wordApp.DisplayAlerts=0
try {
  $docObj=$wordApp.Documents.Open($InputDoc,$false,$true)
  $docObj.Repaginate()
  $docObj.ExportAsFixedFormat($OutputPdf,17)
  Write-Output ('Pages: '+$docObj.ComputeStatistics(2))
  $discardChanges=0
  $docObj.Close([ref]$discardChanges)
} finally {
  $discardChanges=0
  $wordApp.Quit([ref]$discardChanges)
  [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) | Out-Null
}
