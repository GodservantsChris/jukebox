# Using PowerShell, wget is aliased at "Invoke-WebRequest", so use:
# Read each URL and download it
Get-Content .\urls_json_files.txt | ForEach-Object {
    $filename = Split-Path $_ -Leaf
    Invoke-WebRequest -Uri $_ -OutFile $filename
}

Get-Content .\urls_bin_files.txt | ForEach-Object {
    $filename = Split-Path $_ -Leaf
    Invoke-WebRequest -Uri $_ -OutFile $filename
}