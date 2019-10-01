from fiji.analyze.directionality import Directionality_
from ij import WindowManager, ImagePlus
  
# Instantiate plugin
dir = Directionality_()
  
# Set fields and settings
dir.setImagePlus(WindowManager.getCurrentImage())
# dir.setMethod(fiji.analyze.directionality.Directionality_.AnalysisMethod.LOCAL_GRADIENT_ORIENTATION)
dir.setMethod(Directionality_.AnalysisMethod.FOURIER_COMPONENTS)
dir.setBinNumber(30)
dir.setBinStart(-60)
dir.setBuildOrientationMapFlag(True)
  
# Do calculation
dir.computeHistograms()
dir.fitHistograms()
  
# Display plot frame
plot_frame = dir.plotResults()
plot_frame.setVisible(True)
  
# Display fit analysis
data_frame = dir.displayFitAnalysis()
data_frame.setVisible(True) 
  
# Display results table
table = dir.displayResultsTable()
table.show("Directionality histograms")
  
# Display orientation map
stack = dir.getOrientationMap()
ImagePlus("Orientation map", stack).show()
  
# Generate a color wheel
Directionality_.generateColorWheel().show()