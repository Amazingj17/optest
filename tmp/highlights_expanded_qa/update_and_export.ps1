$ErrorActionPreference='Stop'
$filePath='D:\programing\python\optest\docs\项目说明书_亮点详解版.docx'
$wordApp=New-Object -ComObject Word.Application
$wordApp.Visible=$false
$wordApp.DisplayAlerts=0
try {
    $docObj=$wordApp.Documents.Open($filePath,$false,$false)
    try {
        $docObj.Repaginate()
        foreach($toc in $docObj.TablesOfContents){$toc.Update()}
        $docObj.Repaginate()
        foreach($toc in $docObj.TablesOfContents){$toc.UpdatePageNumbers()}
        $docObj.Save()
        $docObj.ExportAsFixedFormat('D:\programing\python\optest\tmp\highlights_expanded_qa\final_word.pdf',17)
        Write-Output "Updated contents; pages=$($docObj.ComputeStatistics(2)); native_equations=$($docObj.OMaths.Count)"
    } finally {$docObj.Close(0)}
} finally {$wordApp.Quit()}
