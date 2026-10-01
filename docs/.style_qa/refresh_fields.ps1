param([string]$InputDoc)
$ErrorActionPreference='Stop'
$wordApp=New-Object -ComObject Word.Application
$wordApp.Visible=$false
$wordApp.DisplayAlerts=0
try {
  $docObj=$wordApp.Documents.Open($InputDoc,$false,$false)
  $docObj.Fields.Update() | Out-Null
  $docObj.Repaginate()
  foreach($tocObj in $docObj.TablesOfContents) { $tocObj.Update() }
  $docObj.Repaginate()
  foreach($tocObj in $docObj.TablesOfContents) { $tocObj.UpdatePageNumbers() }
  $docObj.Save()
  Write-Output ('Pages: '+$docObj.ComputeStatistics(2))
  $discardChanges=0
  $docObj.Close([ref]$discardChanges)
} finally {
  $discardChanges=0
  $wordApp.Quit([ref]$discardChanges)
  [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) | Out-Null
}
