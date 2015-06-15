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
Date                   :=06/15/15
CodeLitePath           :="/home/sawyer/.codelite"
LinkerName             :=/usr/bin/g++-4.9
SharedObjectLinkerName :=/usr/bin/g++-4.9 -shared -fPIC
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
LinkOptions            :=  
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
CXX      := /usr/bin/g++-4.9
CC       := /usr/bin/gcc-4.9
CXXFLAGS :=  -g -O0 -Wall -std=c++11 $(Preprocessors)
CFLAGS   :=  -g -O0 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/AOPotential.cpp$(ObjectSuffix) $(IntermediateDirectory)/force.cpp$(ObjectSuffix) $(IntermediateDirectory)/brownianIntegrator.cpp$(ObjectSuffix) $(IntermediateDirectory)/particle.cpp$(ObjectSuffix) $(IntermediateDirectory)/system.cpp$(ObjectSuffix) $(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) 



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

$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM "main.cpp"

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) "main.cpp"

$(IntermediateDirectory)/AOPotential.cpp$(ObjectSuffix): AOPotential.cpp $(IntermediateDirectory)/AOPotential.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/AOPotential.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/AOPotential.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/AOPotential.cpp$(DependSuffix): AOPotential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/AOPotential.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/AOPotential.cpp$(DependSuffix) -MM "AOPotential.cpp"

$(IntermediateDirectory)/AOPotential.cpp$(PreprocessSuffix): AOPotential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/AOPotential.cpp$(PreprocessSuffix) "AOPotential.cpp"

$(IntermediateDirectory)/force.cpp$(ObjectSuffix): force.cpp $(IntermediateDirectory)/force.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/force.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/force.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/force.cpp$(DependSuffix): force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/force.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/force.cpp$(DependSuffix) -MM "force.cpp"

$(IntermediateDirectory)/force.cpp$(PreprocessSuffix): force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/force.cpp$(PreprocessSuffix) "force.cpp"

$(IntermediateDirectory)/brownianIntegrator.cpp$(ObjectSuffix): brownianIntegrator.cpp $(IntermediateDirectory)/brownianIntegrator.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/brownianIntegrator.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/brownianIntegrator.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/brownianIntegrator.cpp$(DependSuffix): brownianIntegrator.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/brownianIntegrator.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/brownianIntegrator.cpp$(DependSuffix) -MM "brownianIntegrator.cpp"

$(IntermediateDirectory)/brownianIntegrator.cpp$(PreprocessSuffix): brownianIntegrator.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/brownianIntegrator.cpp$(PreprocessSuffix) "brownianIntegrator.cpp"

$(IntermediateDirectory)/particle.cpp$(ObjectSuffix): particle.cpp $(IntermediateDirectory)/particle.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/particle.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/particle.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/particle.cpp$(DependSuffix): particle.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/particle.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/particle.cpp$(DependSuffix) -MM "particle.cpp"

$(IntermediateDirectory)/particle.cpp$(PreprocessSuffix): particle.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/particle.cpp$(PreprocessSuffix) "particle.cpp"

$(IntermediateDirectory)/system.cpp$(ObjectSuffix): system.cpp $(IntermediateDirectory)/system.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/system.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/system.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/system.cpp$(DependSuffix): system.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/system.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/system.cpp$(DependSuffix) -MM "system.cpp"

$(IntermediateDirectory)/system.cpp$(PreprocessSuffix): system.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/system.cpp$(PreprocessSuffix) "system.cpp"

$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix): utilities.cpp $(IntermediateDirectory)/utilities.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/utilities.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/utilities.cpp$(DependSuffix): utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/utilities.cpp$(DependSuffix) -MM "utilities.cpp"

$(IntermediateDirectory)/utilities.cpp$(PreprocessSuffix): utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/utilities.cpp$(PreprocessSuffix) "utilities.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


