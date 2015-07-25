##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Release
ProjectName            :=Clustered
ConfigurationName      :=Release
WorkspacePath          := "/home/sawyer/Programming/PhDResearch"
ProjectPath            := "/home/sawyer/Programming/PhDResearch/Clustered"
IntermediateDirectory  :=./Release
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Sawyer Hopkins
Date                   :=07/25/15
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
Preprocessors          :=$(PreprocessorSwitch)NDEBUG 
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
CXXFLAGS :=  -O2 -Wall $(Preprocessors)
CFLAGS   :=  -O2 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_error.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_main.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/src_runSim.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_analysis.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_force.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_config.cpp$(ObjectSuffix) 



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
	@test -d ./Release || $(MakeDirCommand) ./Release

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix): src/system.cpp $(IntermediateDirectory)/src_system.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/system.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_system.cpp$(DependSuffix): src/system.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_system.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_system.cpp$(DependSuffix) -MM "src/system.cpp"

$(IntermediateDirectory)/src_system.cpp$(PreprocessSuffix): src/system.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_system.cpp$(PreprocessSuffix) "src/system.cpp"

$(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix): src/systemAnalysis.cpp $(IntermediateDirectory)/src_systemAnalysis.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemAnalysis.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemAnalysis.cpp$(DependSuffix): src/systemAnalysis.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_systemAnalysis.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_systemAnalysis.cpp$(DependSuffix) -MM "src/systemAnalysis.cpp"

$(IntermediateDirectory)/src_systemAnalysis.cpp$(PreprocessSuffix): src/systemAnalysis.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemAnalysis.cpp$(PreprocessSuffix) "src/systemAnalysis.cpp"

$(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix): src/systemHandling.cpp $(IntermediateDirectory)/src_systemHandling.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemHandling.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemHandling.cpp$(DependSuffix): src/systemHandling.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_systemHandling.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_systemHandling.cpp$(DependSuffix) -MM "src/systemHandling.cpp"

$(IntermediateDirectory)/src_systemHandling.cpp$(PreprocessSuffix): src/systemHandling.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemHandling.cpp$(PreprocessSuffix) "src/systemHandling.cpp"

$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix): src/systemInit.cpp $(IntermediateDirectory)/src_systemInit.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemInit.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemInit.cpp$(DependSuffix): src/systemInit.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_systemInit.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_systemInit.cpp$(DependSuffix) -MM "src/systemInit.cpp"

$(IntermediateDirectory)/src_systemInit.cpp$(PreprocessSuffix): src/systemInit.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemInit.cpp$(PreprocessSuffix) "src/systemInit.cpp"

$(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix): src/systemOutput.cpp $(IntermediateDirectory)/src_systemOutput.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemOutput.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemOutput.cpp$(DependSuffix): src/systemOutput.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_systemOutput.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_systemOutput.cpp$(DependSuffix) -MM "src/systemOutput.cpp"

$(IntermediateDirectory)/src_systemOutput.cpp$(PreprocessSuffix): src/systemOutput.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemOutput.cpp$(PreprocessSuffix) "src/systemOutput.cpp"

$(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix): src/systemRecovery.cpp $(IntermediateDirectory)/src_systemRecovery.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/systemRecovery.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_systemRecovery.cpp$(DependSuffix): src/systemRecovery.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_systemRecovery.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_systemRecovery.cpp$(DependSuffix) -MM "src/systemRecovery.cpp"

$(IntermediateDirectory)/src_systemRecovery.cpp$(PreprocessSuffix): src/systemRecovery.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_systemRecovery.cpp$(PreprocessSuffix) "src/systemRecovery.cpp"

$(IntermediateDirectory)/src_error.cpp$(ObjectSuffix): src/error.cpp $(IntermediateDirectory)/src_error.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/error.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_error.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_error.cpp$(DependSuffix): src/error.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_error.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_error.cpp$(DependSuffix) -MM "src/error.cpp"

$(IntermediateDirectory)/src_error.cpp$(PreprocessSuffix): src/error.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_error.cpp$(PreprocessSuffix) "src/error.cpp"

$(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix): src/timer.cpp $(IntermediateDirectory)/src_timer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/timer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_timer.cpp$(DependSuffix): src/timer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_timer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_timer.cpp$(DependSuffix) -MM "src/timer.cpp"

$(IntermediateDirectory)/src_timer.cpp$(PreprocessSuffix): src/timer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_timer.cpp$(PreprocessSuffix) "src/timer.cpp"

$(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix): src/utilities.cpp $(IntermediateDirectory)/src_utilities.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/utilities.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_utilities.cpp$(DependSuffix): src/utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_utilities.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_utilities.cpp$(DependSuffix) -MM "src/utilities.cpp"

$(IntermediateDirectory)/src_utilities.cpp$(PreprocessSuffix): src/utilities.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_utilities.cpp$(PreprocessSuffix) "src/utilities.cpp"

$(IntermediateDirectory)/src_main.cpp$(ObjectSuffix): src/main.cpp $(IntermediateDirectory)/src_main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_main.cpp$(DependSuffix): src/main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_main.cpp$(DependSuffix) -MM "src/main.cpp"

$(IntermediateDirectory)/src_main.cpp$(PreprocessSuffix): src/main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_main.cpp$(PreprocessSuffix) "src/main.cpp"

$(IntermediateDirectory)/src_runSim.cpp$(ObjectSuffix): src/runSim.cpp $(IntermediateDirectory)/src_runSim.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/runSim.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_runSim.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_runSim.cpp$(DependSuffix): src/runSim.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_runSim.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_runSim.cpp$(DependSuffix) -MM "src/runSim.cpp"

$(IntermediateDirectory)/src_runSim.cpp$(PreprocessSuffix): src/runSim.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_runSim.cpp$(PreprocessSuffix) "src/runSim.cpp"

$(IntermediateDirectory)/src_analysis.cpp$(ObjectSuffix): src/analysis.cpp $(IntermediateDirectory)/src_analysis.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/analysis.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_analysis.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_analysis.cpp$(DependSuffix): src/analysis.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_analysis.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_analysis.cpp$(DependSuffix) -MM "src/analysis.cpp"

$(IntermediateDirectory)/src_analysis.cpp$(PreprocessSuffix): src/analysis.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_analysis.cpp$(PreprocessSuffix) "src/analysis.cpp"

$(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix): src/particle.cpp $(IntermediateDirectory)/src_particle.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/particle.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_particle.cpp$(DependSuffix): src/particle.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_particle.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_particle.cpp$(DependSuffix) -MM "src/particle.cpp"

$(IntermediateDirectory)/src_particle.cpp$(PreprocessSuffix): src/particle.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_particle.cpp$(PreprocessSuffix) "src/particle.cpp"

$(IntermediateDirectory)/src_force.cpp$(ObjectSuffix): src/force.cpp $(IntermediateDirectory)/src_force.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/force.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_force.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_force.cpp$(DependSuffix): src/force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_force.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_force.cpp$(DependSuffix) -MM "src/force.cpp"

$(IntermediateDirectory)/src_force.cpp$(PreprocessSuffix): src/force.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_force.cpp$(PreprocessSuffix) "src/force.cpp"

$(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix): src/cell.cpp $(IntermediateDirectory)/src_cell.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/cell.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_cell.cpp$(DependSuffix): src/cell.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_cell.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_cell.cpp$(DependSuffix) -MM "src/cell.cpp"

$(IntermediateDirectory)/src_cell.cpp$(PreprocessSuffix): src/cell.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_cell.cpp$(PreprocessSuffix) "src/cell.cpp"

$(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix): src/brownianIntegrator.cpp $(IntermediateDirectory)/src_brownianIntegrator.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/brownianIntegrator.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_brownianIntegrator.cpp$(DependSuffix): src/brownianIntegrator.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_brownianIntegrator.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_brownianIntegrator.cpp$(DependSuffix) -MM "src/brownianIntegrator.cpp"

$(IntermediateDirectory)/src_brownianIntegrator.cpp$(PreprocessSuffix): src/brownianIntegrator.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_brownianIntegrator.cpp$(PreprocessSuffix) "src/brownianIntegrator.cpp"

$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix): src/AOPotential.cpp $(IntermediateDirectory)/src_AOPotential.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/AOPotential.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_AOPotential.cpp$(DependSuffix): src/AOPotential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_AOPotential.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_AOPotential.cpp$(DependSuffix) -MM "src/AOPotential.cpp"

$(IntermediateDirectory)/src_AOPotential.cpp$(PreprocessSuffix): src/AOPotential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_AOPotential.cpp$(PreprocessSuffix) "src/AOPotential.cpp"

$(IntermediateDirectory)/src_config.cpp$(ObjectSuffix): src/config.cpp $(IntermediateDirectory)/src_config.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/sawyer/Programming/PhDResearch/Clustered/src/config.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_config.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_config.cpp$(DependSuffix): src/config.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_config.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_config.cpp$(DependSuffix) -MM "src/config.cpp"

$(IntermediateDirectory)/src_config.cpp$(PreprocessSuffix): src/config.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_config.cpp$(PreprocessSuffix) "src/config.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Release/


