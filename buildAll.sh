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
echo Making Yukawa Potential
tput setaf 7
echo

cd Yukawa
make clean
echo
./Yukawa.sh
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
cp Yukawa/bin/Yukawa.so ClusteredUI/bin/Yukawa.so