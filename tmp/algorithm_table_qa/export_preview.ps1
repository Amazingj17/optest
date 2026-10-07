$ErrorActionPreference = 'Stop'
$qaDir = 'D:\programing\python\optest\tmp\algorithm_table_qa'
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    foreach ($variant in @('before', 'after')) {
        $filePath = if ($variant -eq 'before') { 'D:\programing\python\optest\docs\项目说明书.docx' } else { 'D:\programing\python\optest\docs\项目说明书_算法三线表版.docx' }
        $docObj = $wordApp.Documents.Open($filePath, $false, $true)
        try {
            $docObj.Repaginate()
            $rangeObj = $docObj.Content.Duplicate
            $found = $rangeObj.Find.Execute('清空旧候选')
            $algorithmPage = if ($found) { $rangeObj.Information(3) } else { 0 }
            $count = $docObj.ComputeStatistics(2)
            $docObj.ExportAsFixedFormat((Join-Path $qaDir ($variant + '.pdf')), 17)
            Write-Output "$variant pages=$count algorithm_page=$algorithmPage"
        } finally { $docObj.Close(0) }
    }
} finally { $wordApp.Quit() }
