$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$contentPath = Join-Path $root 'group_meeting_ppt_content_20260816.json'
$content = Get-Content -Raw -Encoding UTF8 $contentPath | ConvertFrom-Json
$templatePath = (Get-ChildItem -LiteralPath $root -Filter '7.19*.pptx' | Where-Object { $_.Name -notlike '~$*' } | Select-Object -First 1).FullName
$outPath = Join-Path $root $content.output_file
$qcImage = Join-Path $root 'data\processed_ctspine1k_real\ctspine1k-msd-t10-liver_0\qc_contact_sheet.png'
$webImage = Join-Path $root '.devspace-computer\reference_ppt_first.png'
function RGBc([int]$r,[int]$g,[int]$b) { return ($r + 256*$g + 65536*$b) }
function Pt([double]$inch) { return [single]($inch * 72.0) }

$NAVY = RGBc 45 71 126
$BLUE = RGBc 63 102 169
$LIGHTBLUE = RGBc 229 236 248
$PALE = RGBc 246 249 253
$TEXT = RGBc 36 49 72
$MUTED = RGBc 99 116 145
$ORANGE = RGBc 197 82 48
$ORANGEPALE = RGBc 252 241 236
$GREEN = RGBc 49 139 98
$RED = RGBc 181 65 54
$WHITE = RGBc 255 255 255
$BORDER = RGBc 220 228 239

function AddText($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [double]$size, [int]$color, [bool]$bold=$false, [int]$align=1) {
    $sh = $slide.Shapes.AddTextbox(1, (Pt $x), (Pt $y), (Pt $w), (Pt $h))
    $sh.TextFrame.TextRange.Text = $text
    $sh.TextFrame.MarginLeft = Pt 0.02
    $sh.TextFrame.MarginRight = Pt 0.02
    $sh.TextFrame.MarginTop = Pt 0.01
    $sh.TextFrame.MarginBottom = Pt 0.01
    $sh.TextFrame.WordWrap = -1
    $sh.TextFrame.TextRange.Font.Name = 'Microsoft YaHei'
    $sh.TextFrame.TextRange.Font.Size = $size
    $sh.TextFrame.TextRange.Font.Color.RGB = $color
    $sh.TextFrame.TextRange.Font.Bold = $(if($bold){-1}else{0})
    $sh.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $sh
}

function AddRoundedCard($slide,[double]$x,[double]$y,[double]$w,[double]$h,[int]$fill,[int]$lineColor=$BORDER) {
    $sh = $slide.Shapes.AddShape(5,(Pt $x),(Pt $y),(Pt $w),(Pt $h))
    $sh.Fill.ForeColor.RGB = $fill
    $sh.Fill.Solid()
    $sh.Line.ForeColor.RGB = $lineColor
    $sh.Line.Weight = 0.8
    return $sh
}

function AddBackground($slide,[bool]$cover=$false) {
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.ForeColor.RGB = $WHITE
    $slide.Background.Fill.Solid()

    $oval1 = $slide.Shapes.AddShape(9,(Pt 8.9),(Pt 4.0),(Pt 6.2),(Pt 4.2))
    $oval1.Fill.ForeColor.RGB = $LIGHTBLUE
    $oval1.Fill.Transparency = 0.55
    $oval1.Line.Visible = 0
    $oval2 = $slide.Shapes.AddShape(9,(Pt -1.6),(Pt 5.15),(Pt 7.8),(Pt 2.8))
    $oval2.Fill.ForeColor.RGB = RGBc 243 246 251
    $oval2.Fill.Transparency = 0.10
    $oval2.Line.Visible = 0

    $barH = $(if($cover){0.58}else{0.16})
    $bar = $slide.Shapes.AddShape(1,0,(Pt (7.5-$barH)),(Pt 13.333),(Pt $barH))
    $bar.Fill.ForeColor.RGB = $NAVY
    $bar.Fill.Solid()
    $bar.Line.Visible = 0
}

function AddHeader($slide,[string]$title,[string]$subtitle,[string]$section) {
    AddText $slide $section 0.62 0.34 2.0 0.28 12 $BLUE $true 1 | Out-Null
    AddText $slide $title 0.62 0.66 12.0 0.48 25 $NAVY $true 1 | Out-Null
    AddText $slide $subtitle 0.64 1.16 11.9 0.42 10.5 $MUTED $false 1 | Out-Null
    $line = $slide.Shapes.AddLine((Pt 0.62),(Pt 1.55),(Pt 12.72),(Pt 1.55))
    $line.Line.ForeColor.RGB = $BORDER
    $line.Line.Weight = 1
}

function AddFooterText($slide,[string]$text) {
    AddText $slide $text 9.2 7.28 3.55 0.16 7.5 $WHITE $false 3 | Out-Null
}

function AddPictureFit($slide,[string]$path,[double]$x,[double]$y,[double]$w,[double]$h) {
    $pic = $slide.Shapes.AddPicture($path,0,-1,(Pt $x),(Pt $y),-1,-1)
    $pic.LockAspectRatio = -1
    $scaleW = (Pt $w) / $pic.Width
    $scaleH = (Pt $h) / $pic.Height
    $scale = [Math]::Min($scaleW,$scaleH)
    $pic.Width = $pic.Width * $scale
    $pic.Height = $pic.Height * $scale
    $pic.Left = (Pt $x) + ((Pt $w)-$pic.Width)/2
    $pic.Top = (Pt $y) + ((Pt $h)-$pic.Height)/2
    return $pic
}

$app = New-Object -ComObject PowerPoint.Application
$app.Visible = -1

try {
    if (Test-Path $outPath) { Remove-Item -Force $outPath }
    $src = $app.Presentations.Open($templatePath,0,0,0)
    $src.SaveCopyAs($outPath)
    $src.Close()

    $pres = $app.Presentations.Open($outPath,0,0,0)
    while ($pres.Slides.Count -gt 0) { $pres.Slides.Item(1).Delete() }
    $layoutBlank = 12

    # Slide 1 - Cover
    $s1 = $pres.Slides.Add(1,$layoutBlank)
    AddBackground $s1 $true
    AddText $s1 'AI MEDICAL - NATIONAL INNOVATION PROJECT' 0.78 0.72 5.5 0.3 11 $BLUE $true 1 | Out-Null
    AddText $s1 $content.cover_title 0.78 1.35 11.7 1.25 28 $NAVY $true 1 | Out-Null
    AddText $s1 $content.cover_subtitle 0.8 2.8 11.2 0.52 12 $MUTED $false 1 | Out-Null

    $status = AddRoundedCard $s1 0.78 3.62 11.7 0.74 $PALE $LIGHTBLUE
    AddText $s1 $content.cover_status 1.04 3.83 11.1 0.32 14 $TEXT $true 1 | Out-Null

    $chipX = 0.8
    foreach($chip in $content.cover_chips) {
        $c = AddRoundedCard $s1 $chipX 4.75 3.45 0.78 $WHITE $BORDER
        $dot = $s1.Shapes.AddShape(9,(Pt ($chipX+0.18)),(Pt 4.98),(Pt 0.22),(Pt 0.22))
        $dot.Fill.ForeColor.RGB = $BLUE; $dot.Line.Visible = 0
        AddText $s1 $chip ($chipX+0.52) 4.92 2.72 0.32 10.5 $TEXT $true 1 | Out-Null
        $chipX += 3.72
    }
    AddText $s1 $content.footer 0.8 7.05 5.2 0.25 9.8 $WHITE $true 1 | Out-Null
    AddText $s1 '2026.08.16' 10.9 7.05 1.55 0.25 9.5 $WHITE $false 3 | Out-Null

    # Slide 2 - Progress overview
    $s2 = $pres.Slides.Add(2,$layoutBlank)
    AddBackground $s2
    AddHeader $s2 $content.slide2_title $content.slide2_subtitle 'PROGRESS OVERVIEW'

    $x = 0.62
    $cardW = 2.38
    $gap = 0.13
    foreach($p in $content.progress) {
        $card = AddRoundedCard $s2 $x 1.84 $cardW 2.25 $WHITE $BORDER
        AddText $s2 $p.name ($x+0.18) 2.02 ($cardW-0.36) 0.34 13 $NAVY $true 1 | Out-Null
        AddText $s2 ([string]$p.pct + '%') ($x+0.18) 2.5 ($cardW-0.36) 0.55 23 $BLUE $true 1 | Out-Null
        $bg = $s2.Shapes.AddShape(5,(Pt ($x+0.18)),(Pt 3.15),(Pt ($cardW-0.36)),(Pt 0.12))
        $bg.Fill.ForeColor.RGB = RGBc 235 239 246; $bg.Line.Visible=0
        $fg = $s2.Shapes.AddShape(5,(Pt ($x+0.18)),(Pt 3.15),(Pt (($cardW-0.36)*$p.pct/100.0)),(Pt 0.12))
        $fg.Fill.ForeColor.RGB = $BLUE; $fg.Line.Visible=0
        AddText $s2 $p.status ($x+0.18) 3.42 ($cardW-0.36) 0.48 8.4 $MUTED $false 1 | Out-Null
        $x += $cardW + $gap
    }

    $statX = 0.72
    foreach($st in $content.slide2_stats) {
        $stat = AddRoundedCard $s2 $statX 4.45 3.35 1.28 $PALE $LIGHTBLUE
        AddText $s2 $st.num ($statX+0.18) 4.64 0.95 0.5 22 $NAVY $true 2 | Out-Null
        AddText $s2 $st.label ($statX+1.12) 4.62 1.95 0.58 9.2 $TEXT $true 1 | Out-Null
        $statX += 3.58
    }

    $call = AddRoundedCard $s2 0.72 6.02 11.9 0.72 $ORANGEPALE (RGBc 234 183 162)
    $accent = $s2.Shapes.AddShape(1,(Pt 0.72),(Pt 6.02),(Pt 0.12),(Pt 0.72))
    $accent.Fill.ForeColor.RGB = $ORANGE; $accent.Line.Visible=0
    AddText $s2 $content.slide2_callout 1.02 6.19 11.2 0.36 11 $TEXT $true 1 | Out-Null
    AddFooterText $s2 $content.footer

    # Slide 3 - Evidence
    $s3 = $pres.Slides.Add(3,$layoutBlank)
    AddBackground $s3
    AddHeader $s3 $content.slide3_title $content.slide3_subtitle 'KEY EVIDENCE'

    $imgCard = AddRoundedCard $s3 0.58 1.79 5.75 4.58 $WHITE $BORDER
    if (Test-Path $qcImage) { AddPictureFit $s3 $qcImage 0.77 2.02 5.37 3.86 | Out-Null }
    AddText $s3 $content.slide3_image1_caption 0.82 5.92 5.25 0.32 8.4 $MUTED $false 2 | Out-Null

    $ry = 1.80
    foreach($g in $content.slide3_groups) {
        $card = AddRoundedCard $s3 6.58 $ry 6.05 1.18 $WHITE $BORDER
        $stripe = $s3.Shapes.AddShape(1,(Pt 6.58),(Pt $ry),(Pt 0.11),(Pt 1.18))
        $stripe.Fill.ForeColor.RGB = $BLUE; $stripe.Line.Visible=0
        AddText $s3 $g.head 6.88 ($ry+0.16) 2.4 0.3 11.5 $NAVY $true 1 | Out-Null
        AddText $s3 $g.body 6.88 ($ry+0.48) 5.42 0.56 8.7 $TEXT $false 1 | Out-Null
        $ry += 1.32
    }

    if (Test-Path $webImage) {
        $webCard = AddRoundedCard $s3 6.58 5.82 2.85 0.86 $PALE $BORDER
        AddPictureFit $s3 $webImage 6.68 5.9 2.65 0.66 | Out-Null
    }
    AddText $s3 $content.slide3_image2_caption 9.58 5.9 2.92 0.5 7.9 $MUTED $false 1 | Out-Null
    AddText $s3 ($content.slide3_badges -join '   /   ') 9.58 6.42 2.92 0.22 7.8 $BLUE $true 1 | Out-Null
    AddFooterText $s3 $content.footer

    # Slide 4 - Risks and next steps
    $s4 = $pres.Slides.Add(4,$layoutBlank)
    AddBackground $s4
    AddHeader $s4 $content.slide4_title $content.slide4_subtitle 'NEXT ACTIONS'

    $leftCard = AddRoundedCard $s4 0.62 1.82 4.18 4.92 $WHITE (RGBc 235 210 201)
    $lh = $s4.Shapes.AddShape(5,(Pt 0.8),(Pt 2.0),(Pt 3.82),(Pt 0.56))
    $lh.Fill.ForeColor.RGB = $ORANGEPALE; $lh.Line.Visible=0
    AddText $s4 $content.slide4_left_head 1.02 2.14 3.35 0.28 12 $ORANGE $true 1 | Out-Null
    $iy = 2.78
    foreach($item in $content.slide4_left_items) {
        $dot = $s4.Shapes.AddShape(9,(Pt 0.94),(Pt ($iy+0.07)),(Pt 0.12),(Pt 0.12))
        $dot.Fill.ForeColor.RGB = $ORANGE; $dot.Line.Visible=0
        AddText $s4 $item 1.18 $iy 3.25 0.58 8.9 $TEXT $false 1 | Out-Null
        $iy += 0.73
    }

    AddText $s4 $content.slide4_right_head 5.18 1.92 3.0 0.36 13 $NAVY $true 1 | Out-Null
    $sy = 2.5
    foreach($step in $content.slide4_steps) {
        $circle = $s4.Shapes.AddShape(9,(Pt 5.2),(Pt $sy),(Pt 0.54),(Pt 0.54))
        $circle.Fill.ForeColor.RGB = $NAVY; $circle.Line.Visible=0
        AddText $s4 $step.n 5.2 ($sy+0.1) 0.54 0.25 8.5 $WHITE $true 2 | Out-Null
        if($step.n -ne '05') {
            $conn = $s4.Shapes.AddLine((Pt 5.47),(Pt ($sy+0.55)),(Pt 5.47),(Pt ($sy+0.93)))
            $conn.Line.ForeColor.RGB = RGBc 188 201 223; $conn.Line.Weight=2
        }
        AddText $s4 $step.t 5.95 ($sy-0.02) 2.28 0.32 11.2 $NAVY $true 1 | Out-Null
        AddText $s4 $step.d 8.05 ($sy-0.02) 4.35 0.52 8.7 $TEXT $false 1 | Out-Null
        $sy += 0.83
    }

    $con = AddRoundedCard $s4 5.18 6.02 7.42 0.72 $PALE $LIGHTBLUE
    AddText $s4 $content.conclusion 5.45 6.14 6.9 0.46 8.9 $NAVY $true 1 | Out-Null
    AddFooterText $s4 $content.footer

    # Basic metadata and save
    $pres.Save()
    $pres.Close()
}
finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}

Write-Output $outPath
