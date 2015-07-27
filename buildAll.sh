#----------

echo
tput setaf 2
echo Making core library
tput setaf 7
echo

cd ClusteredCore
make clean
echo
./ClusteredCore.sh
cd ../

#----------

echo
tput setaf 2
echo Making AOPotential
tput setaf 7
echo

cd AOPotential
make clean
echo
./AOPotential.sh
cd ../

#----------

echo
tput setaf 2
echo Making user interface
tput setaf 7
echo 

cd ClusteredUI
make clean
echo
./ClusteredUI.sh
cd ../

#----------

echo
tput setaf 2
echo Copying Forces
tput setaf 7
echo

cp AOPotential/bin/AOPot.so ClusteredUI/bin/AOPot.so