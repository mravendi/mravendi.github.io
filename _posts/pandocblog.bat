pandoc -o %~n1.md --extract-media=%~n1 %1 -w gfm --atx-headers --columns 9999