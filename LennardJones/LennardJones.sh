
echo Executing Pre Build commands ...
echo
tput setaf 6
echo " Getting updated headers"
cp ../ClusteredCore/include/*.h include/
echo " Getting updated libraries"
cp ../ClusteredCore/bin/*.a linked/
tput setaf 7
echo Done
make
