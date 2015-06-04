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
Date                   :=06/03/15
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
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/point.cpp$(ObjectSuffix) $(IntermediateDirectory)/verlet.cpp$(ObjectSuffix) $(IntermediateDirectory)/force.cpp$(ObjectSuffix) $(IntermediateDirectory)/GnuPlotter.cpp$(ObjectSuffix) $(IntermediateDirectory)/drag.cpp$(ObjectSuffix) $(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) $(IntermediateDirectory)/aggForce.cpp$(ObjectSuffix) $(IntermediateDirectory)/brownianForce.cpp$(ObjectSuffix) 



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

$(IntermediateDirectory)/point.cpp$(ObjectSuffix): point.cpp $(IntermediateDirectory)/point.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/point.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/point.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/point.cpp$(DependSuffix): point.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/point.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/point.cpp$(DependSuffix) -MM "point.cpp"

$(IntermediateDirectory)/point.cpp$(PreprocessSuffix): point.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/point.cpp$(PreprocessSuffix) "point.cpp"

$(IntermediateDirectory)/verlet.cpp$(ObjectSuffix): verlet.cpp $(IntermediateDirectory)/verlet.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/verlet.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/verlet.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/verlet.cpp$(DependSuffix): verlet.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/verlet.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/verlet.cpp$(DependSuffix) -MM "verlet.cpp"

$(IntermediateDirectory)/verlet.cpp$(PreprocessSuffix): verlet.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/verlet.cpp$(PreprocessSuffix) "verlet.cpp"

$(IntermediateDirectory)/force.cpp$(ObjectSuffix): force.cpp $(IntermediateDirectory)/force.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/force.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/force.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/force.cpp$(DependSuffix): force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/force.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/force.cpp$(DependSuffix) -MM "force.cpp"

$(IntermediateDirectory)/force.cpp$(PreprocessSuffix): force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/force.cpp$(PreprocessSuffix) "force.cpp"

$(IntermediateDirectory)/GnuPlotter.cpp$(ObjectSuffix): GnuPlotter.cpp $(IntermediateDirectory)/GnuPlotter.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/GnuPlotter.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/GnuPlotter.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/GnuPlotter.cpp$(DependSuffix): GnuPlotter.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/GnuPlotter.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/GnuPlotter.cpp$(DependSuffix) -MM "GnuPlotter.cpp"

$(IntermediateDirectory)/GnuPlotter.cpp$(PreprocessSuffix): GnuPlotter.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/GnuPlotter.cpp$(PreprocessSuffix) "GnuPlotter.cpp"

$(IntermediateDirectory)/drag.cpp$(ObjectSuffix): drag.cpp $(IntermediateDirectory)/drag.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/drag.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/drag.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/drag.cpp$(DependSuffix): drag.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/drag.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/drag.cpp$(DependSuffix) -MM "drag.cpp"

$(IntermediateDirectory)/drag.cpp$(PreprocessSuffix): drag.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/drag.cpp$(PreprocessSuffix) "drag.cpp"

$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix): utilities.cpp $(IntermediateDirectory)/utilities.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/utilities.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/utilities.cpp$(DependSuffix): utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/utilities.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/utilities.cpp$(DependSuffix) -MM "utilities.cpp"

$(IntermediateDirectory)/utilities.cpp$(PreprocessSuffix): utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/utilities.cpp$(PreprocessSuffix) "utilities.cpp"

$(IntermediateDirectory)/aggForce.cpp$(ObjectSuffix): aggForce.cpp $(IntermediateDirectory)/aggForce.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/aggForce.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/aggForce.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/aggForce.cpp$(DependSuffix): aggForce.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/aggForce.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/aggForce.cpp$(DependSuffix) -MM "aggForce.cpp"

$(IntermediateDirectory)/aggForce.cpp$(PreprocessSuffix): aggForce.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/aggForce.cpp$(PreprocessSuffix) "aggForce.cpp"

$(IntermediateDirectory)/brownianForce.cpp$(ObjectSuffix): brownianForce.cpp $(IntermediateDirectory)/brownianForce.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/brownianForce.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/brownianForce.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/brownianForce.cpp$(DependSuffix): brownianForce.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/brownianForce.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/brownianForce.cpp$(DependSuffix) -MM "brownianForce.cpp"

$(IntermediateDirectory)/brownianForce.cpp$(PreprocessSuffix): brownianForce.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/brownianForce.cpp$(PreprocessSuffix) "brownianForce.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


