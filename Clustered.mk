##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=Clustered
ConfigurationName      :=Debug
WorkspacePath          := "/home/sawyer/Programming/PhDResearch"
ProjectPath            := "/home/sawyer/Programming/PhDResearch/Clustered"
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Sawyer Hopkins
Date                   :=07/11/15
CodeLitePath           :="/home/sawyer/.codelite"
LinkerName             :=/usr/bin/g++-4.8
SharedObjectLinkerName :=/usr/bin/g++-4.8 -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="Clustered.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  -fopenmp
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). 
IncludePCH             := 
RcIncludePath          := 
Libs                   := 
ArLibs                 :=  
LibPath                := $(LibraryPathSwitch). 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++-4.8
CC       := /usr/bin/gcc-4.8
CXXFLAGS :=  -g -O3 -fopenmp -std=c++11 -Wall $(Preprocessors)
CFLAGS   :=  -g -O3 -Wall -fopenmp $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_error.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_force.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_main.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_system.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

PostBuild:
	@echo Executing Post Build commands ...
	rm ./Debug/*.o
	@echo Done

$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix): src/AOPotential.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/AOPotential.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_AOPotential.cpp$(PreprocessSuffix): src/AOPotential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_AOPotential.cpp$(PreprocessSuffix) "src/AOPotential.cpp"

$(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix): src/brownianIntegrator.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/brownianIntegrator.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_brownianIntegrator.cpp$(PreprocessSuffix): src/brownianIntegrator.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_brownianIntegrator.cpp$(PreprocessSuffix) "src/brownianIntegrator.cpp"

$(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix): src/cell.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/cell.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_cell.cpp$(PreprocessSuffix): src/cell.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_cell.cpp$(PreprocessSuffix) "src/cell.cpp"

$(IntermediateDirectory)/src_error.cpp$(ObjectSuffix): src/error.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/error.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_error.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_error.cpp$(PreprocessSuffix): src/error.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_error.cpp$(PreprocessSuffix) "src/error.cpp"

$(IntermediateDirectory)/src_force.cpp$(ObjectSuffix): src/force.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/force.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_force.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_force.cpp$(PreprocessSuffix): src/force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_force.cpp$(PreprocessSuffix) "src/force.cpp"

$(IntermediateDirectory)/src_main.cpp$(ObjectSuffix): src/main.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_main.cpp$(PreprocessSuffix): src/main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_main.cpp$(PreprocessSuffix) "src/main.cpp"

$(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix): src/particle.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/particle.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_particle.cpp$(PreprocessSuffix): src/particle.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_particle.cpp$(PreprocessSuffix) "src/particle.cpp"

$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix): src/system.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/system.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_system.cpp$(PreprocessSuffix): src/system.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_system.cpp$(PreprocessSuffix) "src/system.cpp"

$(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix): src/systemAnalysis.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemAnalysis.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemAnalysis.cpp$(PreprocessSuffix): src/systemAnalysis.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemAnalysis.cpp$(PreprocessSuffix) "src/systemAnalysis.cpp"

$(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix): src/systemHandling.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemHandling.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemHandling.cpp$(PreprocessSuffix): src/systemHandling.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemHandling.cpp$(PreprocessSuffix) "src/systemHandling.cpp"

$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix): src/systemInit.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemInit.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemInit.cpp$(PreprocessSuffix): src/systemInit.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemInit.cpp$(PreprocessSuffix) "src/systemInit.cpp"

$(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix): src/systemOutput.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemOutput.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemOutput.cpp$(PreprocessSuffix): src/systemOutput.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemOutput.cpp$(PreprocessSuffix) "src/systemOutput.cpp"

$(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix): src/systemRecovery.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemRecovery.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemRecovery.cpp$(PreprocessSuffix): src/systemRecovery.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemRecovery.cpp$(PreprocessSuffix) "src/systemRecovery.cpp"

$(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix): src/timer.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/timer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_timer.cpp$(PreprocessSuffix): src/timer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_timer.cpp$(PreprocessSuffix) "src/timer.cpp"

$(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix): src/utilities.cpp 
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/utilities.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_utilities.cpp$(PreprocessSuffix): src/utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_utilities.cpp$(PreprocessSuffix) "src/utilities.cpp"

##
## Clean
##
clean:
	$(RM) -r ./Debug/


