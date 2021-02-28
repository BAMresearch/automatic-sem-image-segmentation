/**
 * MIT License (https://opensource.org/licenses/MIT)
 *
 * Copyright 2020 by the Author Bastian Ruehle
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *  */


import ij.*;
import ij.process.*;
import ij.gui.*;
import ij.plugin.*;
import ij.plugin.filter.*;
import ij.plugin.frame.*;
import ij.measure.*;
import ij.text.*;

import java.io.IOException;
import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Hashtable;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Operation;
import org.tensorflow.Shape;

import java.awt.EventQueue;
import javax.swing.JFrame;
import java.awt.Color;
import java.awt.Dimension;
import javax.swing.JLabel;
import javax.swing.JComboBox;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.ItemListener;
import java.awt.event.ItemEvent;
import javax.swing.JSlider;
import java.awt.Font;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;

import java.awt.GraphicsEnvironment;
import java.awt.GraphicsDevice;

/**
 * Main class implementing the ImageJ PlugIn Class
 * Runs an image 8-Bit through a pre-trained Neural Network (such as a "UNet"), trying create segmentation masks for the particles in the image.
 * Allows some simple post-processing of the raw Neural Network Output. 
 *  */
public class SEM_Particle_Segmentation implements PlugIn {
	private int inputWidth;
	private int inputHeight;
	private String inputNodeName;
	private String outputNodeName;
	private int imageWidth = 0;
	private int imageHeight = 0;
	private int outputChannels = 1;
	private JFrame GUI;
	
	private RoiManager roiMgr = null;
	private ResultsTable resultsTableImage;
	private Roi[] allRois;
	
	private float[] allAreas;
	private float[] allCircularities;
	private float[] allEllipses;
	private float[] allFerets;
	private float[] allMeans;
	private float[] allMedians;
	private float[] allMin;
	private float[] allMax;
	private float[] allPerimeters;
	private float[] allSolidities;
	private float[] allClasses;
	
	private float[] minMaxAllAreas;
	private float[] minMaxAllCircularities;
	private float[] minMaxAllFerets;
	private float[] minMaxAllMeans;
	private float[] minMaxAllPerimeters;
	private float[] minMaxAllSolidities;
	private float[] minMaxAllEllipses;
	private float[] minMaxAllMedians;
	private float[] minMaxAllMin;
	private float[] minMaxAllMax;

	private boolean[] filterAreas;
	private boolean[] filterCircularities;
	private boolean[] filterEllipses;
	private boolean[] filterFerets;
	private boolean[] filterMeans;
	private boolean[] filterMedians;
	private boolean[] filterMin;
	private boolean[] filterMax;
	private boolean[] filterPerimeters;
	private boolean[] filterSolidities;
	private boolean[] filterClasses;

	private String modelSelection;
	private float minThreshold;
	private float minThresholdAutoFilter;
	private boolean applyWatershed;
	Hashtable<String, Path> availableModels = new Hashtable<String, Path>();

	private PlotWindow histPlotWindow;
	private ImagePlus histPlotWindowImp;
	private TextWindow logTextWindow;

	private ImagePlus inputImage;
	private ImagePlus predicted;
	private ImagePlus segmented;
	private ImagePlus classified;

	private static final int roiMgrHeight = 280;
	private static final int roiMgrWidth = 200;

	private static int screenWidth = 1920;
	private static int screenHeight = 1080;
	
	private static final String classificationModelFile = "__ClassificationModel.pb";
	
	/**
	 * Overrides the run method of the PlugIn class.
	 * This method is execute when the plugin is started.
	 *
	 * @param  arg any additional arguments passed to the PlugIn (not used here)
	 * @since 1.8
	 */
	@Override
	public void run(String arg) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice();
					screenWidth = gd.getDisplayMode().getWidth();
					screenHeight = gd.getDisplayMode().getHeight();
					GUI = initializeGUI();
					GUI.setLocation(0, 40);
					GUI.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});		
	}
	
	/**
	 * Creates a Histogram Plot or updates it if one already exists. 
	 *
	 * @param  values {@code float[]} array containing the values used for creating the histogram.
	 * @since 1.8
	 */
	private void updateHistogram(float[] values) {
		double[] x = null;
		double[] y = null;
		double[] y1 = null;
		int count = 0;
		String stats = "";
		String unit = inputImage.getCalibration().getUnit();
		ImageStatistics histStats = null;
		
		if (values.length > 0) {
			FloatProcessor fp = new FloatProcessor(values.length, 1, values);
			// Calculating ImageStatistics directly from the ImageProcessor instead of from ImagePlus doesn't allow choosing Bin Size/Number of Bins
			// ImageStatistics histStats = fp.getStats(); 
	
			ImagePlus imp = new ImagePlus("Histogram (Min Feret) of " + inputImage.getTitle(), fp);
			float[] valMinMax = getArrayMinMax(values);
			int nBins = (int)((valMinMax[1]-valMinMax[0])/inputImage.getCalibration().getX(2.0d)); //use a bin size of 2 pixels (if applicable in calibrated units)
			histStats = imp.getStatistics(ij.measure.Measurements.MEAN | ij.measure.Measurements.MEDIAN | ij.measure.Measurements.STD_DEV, nBins);
	
			// reformat x and y to get "bars" from an xy-Plot
			y1 = histStats.histogram();
			y = new double[2*y1.length];
			x = new double[y.length];
			count = 0;
	
			for (int i = 0; i < y1.length; i++) {
				x[2*i] = histStats.min + i*histStats.binSize - histStats.binSize/2;
				x[2*i+1] = histStats.min + i*histStats.binSize + histStats.binSize/2;
				y[2*i] = y1[i];
				y[2*i+1] = y1[i];
				count += y1[i];
			}

			stats = "N: " + String.valueOf(count) + "\nMean: " + String.format("%.2f", histStats.mean) + " "  + unit + "\nMedian: " + String.format("%.2f", histStats.median)  + " "  + unit + "\nStd.Dev: " + String.format("%.2f", histStats.stdDev) + " "  + unit;
		} else {
			x = new double[1];
			y = new double[1];
			x[0] = 0.0d;
			y[0] = 0.0d;
			stats = "N: 0\nMean: 0.0 " + unit + "\nMedian: 0.0 " + unit + "\nStd.Dev: 0.0 " + unit;
		}

		Plot histPlot = new Plot("Histogram (Min Feret) of " + inputImage.getTitle(), "Min Feret [" + unit + "]", "Count");
		histPlot.addLabel(0.8, 0.05, stats);
		histPlot.setColor("red");
		histPlot.add("filled", x, y);
		if (values.length > 0) {
			histPlot.setLimits(0.0d, histStats.max+histStats.binSize, 0.0d, 1.1d*(double)getArrayMinMax(y1)[1]);
		} else {
			histPlot.setLimits(0.0d, 1.0d, 0.0d, 1.0d);			
		}

		if ((histPlotWindow == null) || (histPlotWindow.isClosed())) {
			histPlotWindow = histPlot.show();
			histPlotWindowImp = histPlotWindow.getImagePlus();
		} else {
			histPlotWindow.drawPlot(histPlot);
		}

		/*
		// Alternative way
		double[] y = new double[values.length];
		for (int i = 0; i < values.length; i++)
		{
			y[i] = (double)values[i];
		}
	  
		histPlot = new Plot("Histogram", "Min Feret", "Count");
		//histPlot.add("filled", histStats.histogram());
		histPlot.addHistogram(y);
		histPlot.show();
		*/

		layoutWindows();
	}

	/**
	 * Applies the Filter Settings on the results of the segmentation, creates the respective overlay, and calls the {@code updateHistogram(float[] values)} method.
	 * Multiple filters are combined with "AND".
	 *
	 * @param  filterName {@code String} identifying which value should be filtered
	 * 		   (can be one of "Min Feret", "Area", "Solidity", "Circularity", or "MeanIntensity")
	 * @param  filterValueLow {@code float} lower bound value of the filter
	 * @param  filterValueHigh {@code float} upper bound value of the filter
	 * @since 1.8
	 */
	private void applyFilterSettings(String filterName, float filterValueLow, float filterValueHigh) {
		Overlay filteredRoisOverlay = new Overlay();
		Overlay allRoisOverlayColored = new Overlay();
		int j = 0;
		for (int i=0; i<allRois.length; i++) {
			if (filterName == "Min Feret") {
				filterFerets[i] = (boolean)(allFerets[i] >= filterValueLow && allFerets[i] <= filterValueHigh);
			}
			if (filterName == "Area") {
				filterAreas[i] = (boolean)(allAreas[i] >= filterValueLow && allAreas[i] <= filterValueHigh);
			}
			if (filterName == "Solidity") {
				filterSolidities[i] = (boolean)(allSolidities[i] >= filterValueLow && allSolidities[i] <= filterValueHigh);
			}
			if (filterName == "Circularity") {
				filterCircularities[i] = (boolean)(allCircularities[i] >= filterValueLow && allCircularities[i] <= filterValueHigh);
			}
			if (filterName == "MeanIntensity") {
				filterMeans[i] = (boolean)(allMeans[i] >= filterValueLow && allMeans[i] <= filterValueHigh);
			}
			if (filterName == "Classes") {
				filterClasses[i] = (boolean)(allClasses[i] >= filterValueLow);
			}
			
			if ((filterFerets[i] && filterAreas[i] && filterSolidities[i] && filterCircularities[i] && filterMeans[i] && filterClasses[i])) {
				filteredRoisOverlay.add(allRois[i]);
				j++;
				allRois[i].setGroup(1);
				allRois[i].setFillColor(new Color(0.0f, 1.0f, 0.0f, 0.25f));
				//allRois[i].setStrokeColor(new Color(0.0f, 1.0f, 1.0f, 1.0f));
				allRoisOverlayColored.add(allRois[i]);
			} else {
				allRois[i].setGroup(2);
				allRois[i].setFillColor(new Color(1.0f, 0.0f, 0.0f, 0.25f));
				//allRois[i].setStrokeColor(new Color(1.0f, 0.0f, 1.0f, 1.0f));
				allRoisOverlayColored.add(allRois[i]);
			}
		}

		inputImage.setOverlay(filteredRoisOverlay);
		resultsTableImage = ResultsTable.getResultsTable();
		resultsTableImage.reset();
		resultsTableImage = filteredRoisOverlay.measure(inputImage);
		resultsTableImage.updateResults();

		inputImage.setOverlay(allRoisOverlayColored);

		if (resultsTableImage.columnExists("MinFeret")) {
			updateHistogram(resultsTableImage.getColumn(resultsTableImage.getColumnIndex("MinFeret")));
		} else {
			ResultsTable.getResultsWindow().getTextPanel().clear();
			updateHistogram(new float[0]);
		}
	}

	/**
	 * Uses the ImageJ Particle Analyzer on a thresholded image. 
	 *
	 * @param  filterName {@code String} identifying which value should be filtered
	 * 		   (can be one of "Min Feret", "Min Area", "Min Solidity", "Min Circularity", or "Min MeanIntensity")
	 * @param  filterValue {@code float} value of the filter
	 * @since 1.8
	 */
	private void doAnalysis() {
		// Do particle Analysis
		if (roiMgr != null) {
			roiMgr.reset();			
		}
		int pAOptions = (ParticleAnalyzer.SHOW_RESULTS | ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES | ParticleAnalyzer.ADD_TO_MANAGER); //ParticleAnalyzer.SHOW_OVERLAY_OUTLINES
		int pAMeasurements = (ij.measure.Measurements.AREA | ij.measure.Measurements.CIRCULARITY | ij.measure.Measurements.ELLIPSE | ij.measure.Measurements.FERET | ij.measure.Measurements.MEAN | ij.measure.Measurements.MEDIAN | ij.measure.Measurements.MIN_MAX | ij.measure.Measurements.PERIMETER);
		ResultsTable resultsTableMask = new ResultsTable();
		double pAMinSize = 0.0d;
		double pAMaxSize = Double.POSITIVE_INFINITY;
		ParticleAnalyzer pA = new ParticleAnalyzer(pAOptions, pAMeasurements, resultsTableMask, pAMinSize, pAMaxSize);
		pA.analyze(segmented);
		roiMgr = RoiManager.getInstance();
		roiMgr.runCommand("Show None");

		roiMgr.runCommand(inputImage, "Measure");
		roiMgr.runCommand("Show None");
		
		resultsTableImage = ResultsTable.getResultsTable();
		allRois = roiMgr.getRoisAsArray().clone();
		if (allRois.length > 0) {
			allAreas = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Area"));
			allCircularities = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Circ."));
			allEllipses = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("AR"));
			allFerets = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("MinFeret"));
			allMeans = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Mean"));
			allMedians = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Median"));
			allMin = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Min"));
			allMax = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Max"));
			allPerimeters = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Perim."));
			allSolidities = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Solidity"));
			allClasses = new float[allAreas.length];
	
			minMaxAllAreas = getArrayMinMax(allAreas);
			minMaxAllCircularities = getArrayMinMax(allCircularities);
			minMaxAllEllipses = getArrayMinMax(allEllipses);
			minMaxAllFerets = getArrayMinMax(allFerets);
			minMaxAllMeans = getArrayMinMax(allMeans);
			minMaxAllMedians = getArrayMinMax(allMedians);
			minMaxAllMin = getArrayMinMax(allMin);
			minMaxAllMax = getArrayMinMax(allMax);
			minMaxAllPerimeters = getArrayMinMax(allPerimeters);
			minMaxAllSolidities = getArrayMinMax(allSolidities);
	
			filterAreas = new boolean[allAreas.length];
			Arrays.fill(filterAreas, true);
			filterCircularities = new boolean[allCircularities.length];
			Arrays.fill(filterCircularities, true);
			filterEllipses = new boolean[allEllipses.length];
			Arrays.fill(filterEllipses, true);
			filterFerets = new boolean[allFerets.length];
			Arrays.fill(filterFerets, true);
			filterMeans = new boolean[allMeans.length];
			Arrays.fill(filterMeans, true);
			filterMedians = new boolean[allMedians.length];
			Arrays.fill(filterMedians, true);
			filterMin = new boolean[allMin.length];
			Arrays.fill(filterMin, true);
			filterMax = new boolean[allMax.length];
			Arrays.fill(filterMax, true);
			filterPerimeters = new boolean[allPerimeters.length];
			Arrays.fill(filterPerimeters, true);
			filterSolidities = new boolean[allSolidities.length];
			Arrays.fill(filterSolidities, true);
			filterClasses = new boolean[allAreas.length];
			Arrays.fill(filterClasses, true);
		}
	}
	
	/**
	 * Runs the active image through the Neural Network for segmentation, following these steps:
	 * 1. Load the Network from a TensorFlow Protobuf file.
	 * 2. Analyze the graph, trying to find the name and shape of the input and output nodes.
	 * 3. Convert the input image in a tensor of appropriate shape, using (overlapping) tiling if necessary.
	 * 4. Run the converted tensor through the Neural Network.
	 * 5. Convert the output tensor back to an image, reassembling individual tiles if necessary.
	 * 6. Runs the {@code segment()} method to threshold (and if desired apply watershed to) the raw output image for further analysis.
	 * 7. Runs the {@code doAnalysis()} method to identify and measure the particles in the thresholded image.
	 * 8. Rearranges some windows to show the results.
	 * 
	 * @since 1.8
	 */
	private void runInference() {		

		if (logTextWindow == null || logTextWindow.isShowing() == false) {
			logTextWindow = new TextWindow("SEM Particle Segmentation Log", "", 350, 300);
		}
		logTextWindow.setLocation(GUI.getLocation().x + GUI.getBounds().width, GUI.getLocation().y);
		logTextWindow.setSize(logTextWindow.getBounds().width, GUI.getBounds().height);
		ij.WindowManager.toFront(ij.WindowManager.getWindow(logTextWindow.getTitle()));

		IJ.showStatus("Running Segmentation...");
		logTextWindow.getTextPanel().append("=============");
		logTextWindow.getTextPanel().append("Segmentation:");
		logTextWindow.getTextPanel().append("=============");
		IJ.showStatus("Loading Model and Analyzing Graph...");
		logTextWindow.getTextPanel().append("Loading Model and Analyzing Graph...");
		byte[] graphDef = readAllBytes(availableModels.get(modelSelection));
		analyzeGraph(graphDef);
		IJ.showStatus("Building Input Tensor from Image for Segmentation...");
		logTextWindow.getTextPanel().append("Building Input Tensor from Image...");
		try (Tensor<Float> inputImageTensor = imageToTensor(inputImage.getProcessor())) { // use try-with-resources to auto-close tensors to prevent resource leaks
			IJ.showStatus("Running Input Tensor through Neural Network...");
			logTextWindow.getTextPanel().append("Running Input Tensor through Neural Network...");
			logTextWindow.getTextPanel().append("...this might take a while...");
			try (Tensor<Float> outputImage = executeGraph(graphDef, inputImageTensor, inputNodeName, outputNodeName)) {
				// Create image from Neural Network Output Tensor
				IJ.showStatus("Creating Image from Output Tensor...");
				logTextWindow.getTextPanel().append("Creating Image from Output Tensor...");
				predicted = tensorToImage(outputImage, 32, 2);
				predicted.setCalibration(inputImage.getCalibration());
				predicted.show();
				// Do thresholding and watershed of predicted Image
				IJ.showStatus("Post-Processing Output Image...");
				logTextWindow.getTextPanel().append("Post-Processing Output Image...");
				segmented = segment();
				segmented.setCalibration(inputImage.getCalibration());
				segmented.show();
				// Do particle Analysis
				IJ.showStatus("Analyzing Particles...");
				logTextWindow.getTextPanel().append("Analyzing Particles...");
				doAnalysis();
				applyFilterSettings("Circularity", 0.0f, 1.0f);
				// Layout Windows
				//layoutWindows();

				IJ.showStatus("Done with Segmentation");
				logTextWindow.getTextPanel().append("Done with Segmentation");
			}
		}
	}

	/**
	 * Applies a threshold (and, if desired, a watershed) to the output image from the Neural Network (which usually contains values between {@code 0} and {@code 1}).
	 * 
	 * @return {@code ImagePlus} object of the processed image
	 * @since 1.8
	 */
	private ImagePlus segment() {
		ImagePlus segmented = IJ.createImage("Segmentation of " + inputImage.getTitle(), imageWidth, imageHeight, 0, 8);
		ImageProcessor ip = predicted.getProcessor().duplicate();
		ip.setThreshold(minThreshold, 1.0d, ImageProcessor.BLACK_AND_WHITE_LUT);
		ImageProcessor ipBin = ip.createMask();
		if (applyWatershed) {
			new EDM().toWatershed(ipBin);
		}
		ipBin.invertLut();
		segmented.setProcessor(ipBin);
		return segmented;
	}

	/**
	 * Runs the active image through the Neural Network for classification, following these steps:
	 * 1. Load the Network from a TensorFlow Protobuf file.
	 * 2. Analyze the graph, trying to find the name and shape of the input and output nodes.
	 * 3. Convert the input image in a tensor of appropriate shape, using (overlapping) tiling if necessary.
	 * 4. Run the converted tensor through the Neural Network.
	 * 5. Convert the output tensor back to an image, reassembling individual tiles if necessary.
	 * 6. Runs the {@code doAnalysis()} method to identify and measure the particles in the thresholded image.
	 * 7. Rearranges some windows to show the results.
	 * 
	 * @return {@code ImagePlus} object of the processed image
	 * @since 1.8
	 */
	private void runAutoFilter() {		

		if (logTextWindow == null || logTextWindow.isShowing() == false) {
			logTextWindow = new TextWindow("SEM Particle Segmentation Log", "", 350, 300);
		}
		logTextWindow.setLocation(GUI.getLocation().x + GUI.getBounds().width, GUI.getLocation().y);
		logTextWindow.setSize(logTextWindow.getBounds().width, GUI.getBounds().height);
		ij.WindowManager.toFront(ij.WindowManager.getWindow(logTextWindow.getTitle()));

		IJ.showStatus("Running Classification...");
		logTextWindow.getTextPanel().append("=============");
		logTextWindow.getTextPanel().append("Classification:");
		logTextWindow.getTextPanel().append("=============");
		IJ.showStatus("Loading Model and Analyzing Graph...");
		logTextWindow.getTextPanel().append("Loading Model and Analyzing Graph...");
		byte[] graphDef = readAllBytes(Paths.get(availableModels.get(modelSelection).getParent().toString(), classificationModelFile));
		analyzeGraph(graphDef);
		IJ.showStatus("Building Input Tensor from Image for Classification...");
		logTextWindow.getTextPanel().append("Building Input Tensor from Image...");
		ImageProcessor ipSeg = segmented.getProcessor().duplicate();
		ipSeg.invertLut();
		try (Tensor<Float> inputImageTensor = imageToTensor(new ImageProcessor[] {inputImage.getProcessor(), ipSeg})) { // use try-with-resources to auto-close tensors to prevent resource leaks
			IJ.showStatus("Running Input Tensor through Neural Network...");
			logTextWindow.getTextPanel().append("Running Input Tensor through Neural Network...");
			logTextWindow.getTextPanel().append("...this might take a while...");
			try (Tensor<Float> outputImage = executeGraph(graphDef, inputImageTensor, inputNodeName, outputNodeName)) {
				// Create image from Neural Network Output Tensor
				IJ.showStatus("Creating Image from Output Tensor...");
				logTextWindow.getTextPanel().append("Creating Image from Output Tensor...");
				classified = tensorToImage(outputImage, 24, 2);
				classified.setCalibration(inputImage.getCalibration());
				classified.show();
				// Apply Filter Settings according to Particle Class
				IJ.showStatus("Classifying and filtering individual particles...");
				logTextWindow.getTextPanel().append("Classifying and filtering individual particles...");

				roiMgr.runCommand(classified, "Measure");
				roiMgr.runCommand("Show None");
				resultsTableImage = ResultsTable.getResultsTable();
				allRois = roiMgr.getRoisAsArray().clone();
				if (allRois.length > 0) {
					allMeans = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Mean"));
					for (int i = 0; i < allMeans.length; i++) {
						allClasses[i] = allMeans[i];
					}
				}
				roiMgr.runCommand(inputImage, "Measure");

				applyFilterSettings("Classes", minThresholdAutoFilter, minThresholdAutoFilter);
				// Layout Windows
				//layoutWindows();

				IJ.showStatus("Done with Auto-Filtering");
				logTextWindow.getTextPanel().append("Done with Auto-Filtering");
			}
		}
	}

	/**
	 * Helper Method that converts an image to a Float Array
	 * If the image is larger than the dimension of the input tensor the Neural Network accepts,
	 * a batch of arrays of appropriate size is created, using overlapping tiles to avoid seams.
	 * 
	 * @param  ip {@code ImageProcessor} containing the pixel data of the input image
	 * @return {@code Float[][][][]} array of the (tiled) pixel data, converted to {@code float} and normalized to the range {@code [0, 1]}, using CHANNELS_LAST
	 * @since 1.8
	 */
	private float[][][][] imageToFloatArray(ImageProcessor ip) {
		int noOfXTiles = (int)Math.ceil((float)imageWidth/inputWidth);
		int noOfYTiles = (int)Math.ceil((float)imageHeight/inputHeight);
		// If more than 1 tile has to be used introduce at least 2 pixel overlap between tiles to avoid seams:
		if ((noOfXTiles > 1) && ((inputWidth - (imageWidth % inputWidth)) % inputWidth <= 2)) {
			noOfXTiles += 1;
		}
		if ((noOfYTiles > 1) && ((inputHeight - (imageHeight % inputHeight))	% inputHeight <= 2)) {
			noOfYTiles += 1;
		}
		int noOfTiles = noOfXTiles*noOfYTiles;
		float[][][][] imageBytes = new float[noOfTiles][inputWidth][inputHeight][1];
		int minValue = 255;
		int maxValue = 0;
		int curPix = 0;
		// Get min and max pixel values
		for (int x = 0; x < imageWidth; x++) {
			for (int y = 0; y < imageHeight; y++) {
				curPix = (int)ip.get(x, y);
				minValue = Math.min(curPix, minValue);
				maxValue = Math.max(curPix, maxValue);
			}
		}
		// Tile Image if it is bigger than the input Tensor (use overlapping tiles), convert to Float, and normalize to [0,1]
		int k = 0;
		int offsetX = 0;
		int offsetY = 0;
		for (int i = 0; i < noOfXTiles; i++) {
			if (noOfXTiles > 1) {
				offsetX = (int)Math.ceil(i*(inputWidth - ((inputWidth*noOfXTiles - imageWidth)/(noOfXTiles-1))));
			} else {
				offsetX = 0;
			}
			for (int j = 0; j < noOfYTiles; j++) {
				if (noOfYTiles > 1) {
					offsetY = (int)Math.ceil(j*(inputHeight - ((inputHeight*noOfYTiles - imageHeight)/(noOfYTiles-1))));
				} else {
					offsetY = 0;
				}
				for (int x = 0; x < inputWidth; x++) {
					for (int y = 0; y < inputHeight; y++) {
							if (x + offsetX >= imageWidth || y + offsetY>= imageHeight) { // if input Tensor is larger than image fill with 0
								imageBytes[k][x][y][0] = 0f;
							} else {
							imageBytes[k][x][y][0] = (float)((ip.getPixelValue(x + offsetX, y + offsetY)-minValue)/((float)(maxValue-minValue)));
							}
					}
				}
				k++;
			}
		}
		return imageBytes;
	}

	/**
	 * Overloaded Method that takes several input images (e.g. for classification)
	 * Converts an image to a TensorFlow tensor that can be run through the Neural Network.
	 * If the image is larger than the dimension of the input tensor the Neural Network accepts,
	 * a batch of tensors of appropriate size is created, using overlapping tiles to avoid seams.
	 * 
	 * @param  ip {@code ImageProcessor} containing the pixel data of the input image
	 * @return {@code Tensor<Float>} object of the (tiled) pixel data, converted to {@code float} and normalized to the range {@code [0, 1]}, using CHANNELS_LAST
	 * @since 1.8
	 */
	private Tensor<Float> imageToTensor(ImageProcessor[] ips) {
		float[][][][] channel;
		float[][][][] imageBytes = null;
		for (int i = 0; i < ips.length; i++) {
			channel = imageToFloatArray(ips[i]);
			if (imageBytes == null) {
				imageBytes = new float[channel.length][channel[0].length][channel[0][0].length][ips.length];
			}
			for (int j = 0; j < imageBytes.length; j++) {
				for (int k = 0; k < imageBytes[0].length; k++) {
					for (int l = 0; l < imageBytes[0][0].length; l++) {
						imageBytes[j][k][l][i] = channel[j][k][l][0];
					}
				}
			}		
		}
		return Tensor.create(imageBytes, Float.class);
	}

	/**
	 * Overloaded Method that takes a single input image (e.g. for segmentation)
	 * Converts an image to a TensorFlow tensor that can be run through the Neural Network.
	 * If the image is larger than the dimension of the input tensor the Neural Network accepts,
	 * a batch of tensors of appropriate size is created, using overlapping tiles to avoid seams.
	 * 
	 * @param  ip {@code ImageProcessor} containing the pixel data of the input image
	 * @return {@code Tensor<Float>} object of the (tiled) pixel data, converted to {@code float} and normalized to the range {@code [0, 1]}, using CHANNELS_LAST
	 * @since 1.8
	 */
	private Tensor<Float> imageToTensor(ImageProcessor ip) {
		return (Tensor.create(imageToFloatArray(ip), Float.class));
	}

	/**
	 * Converts a TensorFlow tensor back to an image.
	 * If the tensor contains a batch of tiles, they are reassembled into a seamless image
	 * by taking the maximum, averaging, or cropping overlapping areas.
	 * 
	 * @param  result {@code Tensor<Float>} object to be converted to an image (assuming CHANNELS_LAST)
	 * @param  bitDepth {@code int} bit depth of the resulting image (32 for raw output, 24 for classification results)
	 * @param  manageOverlapMode {@code int} 0: maximum, 1: average, 2: crop)
	 * @return {@code ImagePlus} object containing the data from the input tensor as pixel data.
	 * @since 1.8
	 */
	private ImagePlus tensorToImage(Tensor<Float> result, int bitDepth, int manageOverlapMode) {
		ImagePlus predicted = null;
		float[][][][] outputImage = result.copyTo(new float[(int)result.shape()[0]][(int)result.shape()[1]][(int)result.shape()[2]][(int)result.shape()[3]]);
		int noOfXTiles = (int)Math.ceil((float)imageWidth/inputWidth);
		int noOfYTiles = (int)Math.ceil((float)imageHeight/inputHeight);
		float curVal = 0.f;
		int overlapSizeX = 0;
		int overlapSizeY = 0;
		if (noOfXTiles > 1) {
			overlapSizeX = (int)((inputWidth * noOfXTiles - imageWidth) / (2.0*(noOfXTiles - 1)));
		}
		if (noOfYTiles > 1) {
			overlapSizeY = (int)((inputHeight * noOfYTiles - imageHeight) / (2.0*(noOfYTiles - 1)));
		}
		int cxl = 0; // left crop
		int cxr = 0; // right crop
		int cyt = 0; // top crop
		int cyb = 0; // bottom crop
		
		// If more than 1 tile has to be used introduce at least 2 pixel overlap between tiles to avoid seams:
		if ((noOfXTiles > 1) && ((inputWidth - (imageWidth % inputWidth)) % inputWidth <= 2)) {
			noOfXTiles += 1;
		}
		if ((noOfYTiles > 1) && ((inputHeight - (imageHeight % inputHeight)) % inputHeight <= 2)) {
			noOfYTiles += 1;
		}
		float[][] pix = new float[imageWidth][imageHeight];
		int[][] overlaps = new int[imageWidth][imageHeight]; // keep track of overlapping tiles
		int k = 0;
		int offsetX = 0;
		int offsetY = 0;
		for (int i = 0; i < noOfXTiles; i++) {
			if (noOfXTiles > 1) {
				offsetX = (int)Math.ceil(i*(inputWidth - ((inputWidth*noOfXTiles - imageWidth)/(noOfXTiles-1))));
			} else {
				offsetX = 0;
			}
			for (int j = 0; j < noOfYTiles; j++) {
				if (noOfYTiles > 1) {
					offsetY = (int)Math.ceil(j*(inputHeight - ((inputHeight*noOfYTiles - imageHeight)/(noOfYTiles-1))));
				} else {
					offsetY = 0;
				}
				for (int x = 1; x < inputWidth-1; x++) { // Discard "border pixels" (start at 1, finish at inputWidth -1)
					for (int y = 1; y < inputHeight-1; y++) {	// Discard "border pixels" (start at 1, finish at inputHeight -1)
						if (x + offsetX >= imageWidth || y + offsetY >= imageHeight) { // if input Tensor is larger than image ignore output
							continue;
						} else {
							if (bitDepth == 32) {
								curVal = (float)(outputImage[k][x][y][this.outputChannels]);
							} else {
								curVal = (float)(0.1*outputImage[k][x][y][0]+0.9*outputImage[k][x][y][1]+0.5*outputImage[k][x][y][2]);
							}
							if (manageOverlapMode == 0) {  // Take maximum in overlapping regions
								pix[x + offsetX][y + offsetY] = Math.max(curVal, pix[x + offsetX][y + offsetY]);
							} else if (manageOverlapMode == 1) {  // Average overlapping regions
								pix[x + offsetX][y + offsetY] += curVal;
								overlaps[x + offsetX][y + offsetY] += 1;
							} else if (manageOverlapMode == 2) { // Crop overlapping regions
								if ((i == 0) && (x > inputWidth-overlapSizeX)){
									x += overlapSizeX-1;
									break;
								} else if ((i == noOfXTiles - 1) && (x < overlapSizeX)) {
									x += overlapSizeX-1;
									break;
								} else if (((i > 0) && (i < noOfXTiles - 1)) && ((x < overlapSizeX) || (x > inputWidth-overlapSizeX))){
									x += overlapSizeX-1;
									break;
								}
								if ((j == 0) && (y > inputHeight-overlapSizeY)){
									y += overlapSizeY-1;
									continue;
								} else if ((j == noOfYTiles - 1) && (y < overlapSizeY)) {
									y += overlapSizeY-1;
									continue;
								} else if (((j > 0) && (j < noOfYTiles - 1)) && ((y < overlapSizeY) || (y > inputHeight-overlapSizeY))){
									y += overlapSizeY-1;
									continue;
								}
								pix[x + offsetX][y + offsetY] = curVal;
							}
						}
					}
				}
				k++;
			}
		}
		
		if (manageOverlapMode == 1) {
			// Average overlapping regions
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
						pix[x][y] /= (float)overlaps[x][y];
				}
			}
		}
		
		if (bitDepth == 32) {
			predicted = IJ.createImage("Raw Output of " + inputImage.getTitle(), imageWidth, imageHeight, 0, 32);		
		} else {
			predicted = IJ.createImage("Classification Results of " + inputImage.getTitle(), imageWidth, imageHeight, 0, 32);					
		}
		predicted.getProcessor().setFloatArray(pix);
		return predicted;
	}

	/**
	 * Constructs a TensorFlow graph from a Protobuf file and analyzes it, trying to find the name and shape of the input and output nodes.
	 * 
	 * @param  graphDef {@code byte[]} array containing the contents of a TensorFlow Protobuf file
	 * @since 1.8
	 */
	private void analyzeGraph(byte[] graphDef) {
		try (Graph g = new Graph()) {
			String o = "";
			Operation op = null;
			g.importGraphDef(graphDef);
			Iterator<Operation> operationIterator = g.operations();
			Shape opShape = null;
			while (operationIterator.hasNext()) {
				op = operationIterator.next();
				if (op.type().equals("Placeholder")) {
					opShape = op.output(0).shape();
					inputWidth = (int)opShape.size(1);
					inputHeight = (int)opShape.size(1);
					inputNodeName = op.name();
				} else if (op.type().equals("Sigmoid") && op.output(0).shape().toString().equals(opShape.toString())) {
					o = op.name();
					this.outputNodeName = op.name();
					this.outputChannels = 0;
				} else if (op.type().equals("Softmax")){					
					o = op.name();
					this.outputNodeName = op.name();
					this.outputChannels = 1;
				}
			}
			if (o == "") {
				o = op.name();
				this.outputNodeName = op.name();
				this.outputChannels = 1;
				logTextWindow.getTextPanel().append("Output layer could not be detected. Using last layer as output: " + o);
			}
		}
	}
	
	/**
	 * Runs a tensor through a neural network and returns the output tensor.
	 * 
	 * @param  graphDef {@code byte[]} array containing the contents of a TensorFlow Protobuf file
	 * @param  image {@code Tensor<Float>} tensor containing the image pixel data
	 * @param  inputNodeName {@code String} specifying the name of the input node
	 * @param  outputNodeName {@code String} specifying the name of the output node
	 * @return {@code Tensor<Float>} object containing output data
	 * @since 1.8
	 */
	public static Tensor<Float> executeGraph(byte[] graphDef, Tensor<Float> image, String inputNodeName, String outputNodeName) {
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g)) {
				Tensor<Float> result = s.runner().feed(inputNodeName, image).fetch(outputNodeName).run().get(0).expect(Float.class); //Model9: fetch("decoded/Sigmoid"); Model10: fetch("activation_57/Sigmoid") or "batch_normalization_85/FusedBatchNorm_1"
				return result;
			}
		}
	}

	/**
	 * Rearranges the windows on the screen to show all relevant outputs
	 * 
	 * @since 1.8
	 */
	private void layoutWindows() {
		int inputImageTargetHeight = (int)(0.667*screenHeight);
		ij.WindowManager.toFront(ij.WindowManager.getWindow(GUI.getTitle()));
		ij.WindowManager.toFront(ij.WindowManager.getWindow(logTextWindow.getTitle()));
		ij.WindowManager.toFront(ij.WindowManager.getWindow(histPlotWindowImp.getTitle()));			
		ij.WindowManager.toFront(ij.WindowManager.getWindow(roiMgr.getTitle()));
		ij.WindowManager.toFront(ij.WindowManager.getWindow(ResultsTable.getResultsTable().getTitle()));
		ij.WindowManager.toFront(ij.WindowManager.getWindow(inputImage.getTitle()));
		GUI.setLocation(0, 40);
		histPlotWindowImp.getWindow().setLocation(GUI.getLocation().x, GUI.getLocation().y + GUI.getBounds().height);
		histPlotWindow.getPlot().setSize(histPlotWindow.getBounds().width, inputImageTargetHeight + roiMgrHeight - GUI.getBounds().height - (histPlotWindow.getBounds().height - histPlotWindow.getPlot().getSize().height));
		inputImage.getWindow().setLocation(GUI.getLocation().x + histPlotWindowImp.getWindow().getBounds().width, GUI.getLocation().y);
		inputImage.getWindow().setSize((int)(inputImage.getWindow().getBounds().width/(float)inputImage.getWindow().getBounds().height*inputImageTargetHeight), inputImageTargetHeight);
		inputImage.getWindow().getCanvas().fitToWindow();
		logTextWindow.setLocation(GUI.getLocation().x + GUI.getBounds().width, GUI.getLocation().y);
		logTextWindow.setSize(histPlotWindowImp.getWindow().getBounds().width - GUI.getBounds().width, GUI.getBounds().height) ;
		roiMgr.setLocation(inputImage.getWindow().getLocation().x + inputImage.getWindow().getBounds().width - roiMgr.getBounds().width, inputImage.getWindow().getLocation().y + inputImage.getWindow().getBounds().height);
		roiMgr.setSize(roiMgrWidth, roiMgrHeight);
		resultsTableImage.getResultsWindow().setLocation(inputImage.getWindow().getLocation().x, inputImage.getWindow().getLocation().y + inputImage.getWindow().getBounds().height);
		resultsTableImage.getResultsWindow().setSize(inputImage.getWindow().getBounds().width - roiMgr.getBounds().width, roiMgr.getBounds().height);
	}

	/**
	 * Reads all the content of a file and returns it as a byte array.
	 * 
	 * @param  path {@code Path} object pointing to the file to be read
	 * @return {@code byte[]} array of the file content
	 * @since 1.8
	 */
	public static byte[] readAllBytes(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
		}
		return null;
	}

	/**
	 * Returns an array containing the minimum and maximum value of a float input array.
	 * 
	 * @param  a {@code float[]} array from which the minimum and maximum values should be obtained
	 * @return {@code float[]} array of the minimum and maximum values in the form [min, max] 
	 * @since 1.8
	 */
	public static float[] getArrayMinMax(float[] a) {
		float minVal = a[0];
		float maxVal = a[0];
		float curVal = a[0];
		// Get min and max values
		for (int i = 0; i < a.length; i++) {
			curVal = a[i];
			minVal = Math.min(curVal, minVal);
			maxVal = Math.max(curVal, maxVal);
		}
		float[] b = {minVal, maxVal};
		return b;
	}

	/**
	 * Returns an array containing the minimum and maximum value of a double input array.
	 * 
	 * @param  a {@code double[]} array from which the minimum and maximum values should be obtained
	 * @return {@code double[]} array of the minimum and maximum values in the form [min, max] 
	 * @since 1.8
	 */
	public double[] getArrayMinMax(double[] a) {
		double minVal = a[0];
		double maxVal = a[0];
		double curVal = a[0];
		// Get min and max values
		for (int i = 0; i < a.length; i++) {
			curVal = a[i];
			minVal = Math.min(curVal, minVal);
			maxVal = Math.max(curVal, maxVal);
		}
		double[] b = {minVal, maxVal};
		return b;
	}
		
	/**
	 * Constructs the GUI as a JFrame and displays it
	 * 
	 * @since 1.8
	 */	
	private JFrame initializeGUI() {
		double nRows = 14.d;
		JFrame frmSemParticleSegmentation = new JFrame();
		//frmSemParticleSegmentation.setLocationByPlatform(true);
		frmSemParticleSegmentation.getContentPane().setSize(new Dimension(288, 420));
		frmSemParticleSegmentation.setName("GUI_Frame");
		frmSemParticleSegmentation.setTitle("SEM Particle Segmentation");
		frmSemParticleSegmentation.setBounds(0, 0, 421, 496);
		frmSemParticleSegmentation.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frmSemParticleSegmentation.setResizable(false);
		GridBagLayout gridBagLayout = new GridBagLayout();
		gridBagLayout.columnWidths = new int[]{10, 167, 167, 10, 0};
		gridBagLayout.rowHeights = new int[]{10, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 16, 10};
		gridBagLayout.columnWeights = new double[]{0.0, 0.0, 1.0, 0.0, Double.MIN_VALUE};
		gridBagLayout.rowWeights = new double[]{0.0, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 1/nRows, 0.0, Double.MIN_VALUE};
		frmSemParticleSegmentation.getContentPane().setLayout(gridBagLayout);
		
		JLabel lblNewLabel = new JLabel("Neural Network Model:");
		lblNewLabel.setFont(new Font("Tahoma", Font.BOLD, 11));
		GridBagConstraints gbc_lblNewLabel = new GridBagConstraints();
		gbc_lblNewLabel.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel.gridx = 1;
		gbc_lblNewLabel.gridy = 1;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel, gbc_lblNewLabel);
		
		JComboBox<String> comboBox = new JComboBox<String>();
		File dir = Paths.get(System.getProperty("user.dir"), "plugins", "SEM_Particle_Segmentation_Models").toFile();
		File[] allModels = dir.listFiles((d, name) -> name.endsWith(".pb") && !name.startsWith("__"));
		String[] modelNames = new String[allModels.length];
		int i = 0;
		for (File f : allModels) {
			availableModels.put(f.getName().replace(".pb", "").replace("_", " "), Paths.get(f.getAbsolutePath()));
			modelNames[i] = f.getName().replace(".pb", "").replace("_", " ");
			i++;
		}		
		comboBox.setBackground(Color.WHITE);
		comboBox.setModel(new DefaultComboBoxModel<String>(modelNames));
		comboBox.setToolTipText("Choose the pre-trained Model that is used for the segmentation");
		GridBagConstraints gbc_comboBox = new GridBagConstraints();
		gbc_comboBox.gridwidth = 2;
		gbc_comboBox.fill = GridBagConstraints.BOTH;
		gbc_comboBox.insets = new Insets(0, 0, 5, 5);
		gbc_comboBox.gridx = 1;
		gbc_comboBox.gridy = 2;
		frmSemParticleSegmentation.getContentPane().add(comboBox, gbc_comboBox);
		
		JLabel lblNewLabel_1 = new JLabel("Options:");
		lblNewLabel_1.setFont(new Font("Tahoma", Font.BOLD, 11));
		GridBagConstraints gbc_lblNewLabel_1 = new GridBagConstraints();
		gbc_lblNewLabel_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1.gridx = 1;
		gbc_lblNewLabel_1.gridy = 3;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_1, gbc_lblNewLabel_1);
		
		JLabel lblNewLabel_3 = new JLabel("");
		GridBagConstraints gbc_lblNewLabel_3 = new GridBagConstraints();
		gbc_lblNewLabel_3.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_3.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_3.gridx = 2;
		gbc_lblNewLabel_3.gridy = 3;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_3, gbc_lblNewLabel_3);
		
		JLabel lblNewLabel_2 = new JLabel("Threshold Value:");
		GridBagConstraints gbc_lblNewLabel_2 = new GridBagConstraints();
		gbc_lblNewLabel_2.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2.gridx = 1;
		gbc_lblNewLabel_2.gridy = 4;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2, gbc_lblNewLabel_2);
		
		JSlider sliderThreshold = new JSlider();
		GridBagConstraints gbc_sliderThreshold = new GridBagConstraints();
		gbc_sliderThreshold.fill = GridBagConstraints.BOTH;
		gbc_sliderThreshold.insets = new Insets(0, 0, 5, 5);
		gbc_sliderThreshold.gridx = 2;
		gbc_sliderThreshold.gridy = 4;
		frmSemParticleSegmentation.getContentPane().add(sliderThreshold, gbc_sliderThreshold);
		
		JLabel lblNewLabel_2_1 = new JLabel("Watershed:");
		GridBagConstraints gbc_lblNewLabel_2_1 = new GridBagConstraints();
		gbc_lblNewLabel_2_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_1.gridx = 1;
		gbc_lblNewLabel_2_1.gridy = 5;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_1, gbc_lblNewLabel_2_1);
		
		JCheckBox chckbxNewCheckBox = new JCheckBox("Apply Watershed");
		chckbxNewCheckBox.setToolTipText("Apply Watershed Algorithm after thresholding for further Particle separation");
		chckbxNewCheckBox.setSelected(true);
		GridBagConstraints gbc_chckbxNewCheckBox = new GridBagConstraints();
		gbc_chckbxNewCheckBox.anchor = GridBagConstraints.WEST;
		gbc_chckbxNewCheckBox.fill = GridBagConstraints.VERTICAL;
		gbc_chckbxNewCheckBox.insets = new Insets(0, 0, 5, 5);
		gbc_chckbxNewCheckBox.gridx = 2;
		gbc_chckbxNewCheckBox.gridy = 5;
		frmSemParticleSegmentation.getContentPane().add(chckbxNewCheckBox, gbc_chckbxNewCheckBox);
		
		JLabel lblNewLabel_1_1 = new JLabel("Filters:");
		lblNewLabel_1_1.setFont(new Font("Tahoma", Font.BOLD, 11));
		lblNewLabel_1_1.setToolTipText("");
		GridBagConstraints gbc_lblNewLabel_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_1_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1.gridx = 1;
		gbc_lblNewLabel_1_1.gridy = 6;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_1_1, gbc_lblNewLabel_1_1);
		
		JLabel lblNewLabel_3_1 = new JLabel("");
		GridBagConstraints gbc_lblNewLabel_3_1 = new GridBagConstraints();
		gbc_lblNewLabel_3_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_3_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_3_1.gridx = 2;
		gbc_lblNewLabel_3_1.gridy = 6;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_3_1, gbc_lblNewLabel_3_1);
		
		RangeSlider sliderArea = new RangeSlider();
		sliderArea.setValue(0);
		sliderArea.setUpperValue(100);
		sliderArea.setEnabled(true);
		GridBagConstraints gbc_sliderArea = new GridBagConstraints();
		gbc_sliderArea.fill = GridBagConstraints.BOTH;
		gbc_sliderArea.insets = new Insets(0, 0, 5, 5);
		gbc_sliderArea.gridx = 2;
		gbc_sliderArea.gridy = 8;
		frmSemParticleSegmentation.getContentPane().add(sliderArea, gbc_sliderArea);
		
		RangeSlider sliderMinFeret = new RangeSlider();
		sliderMinFeret.setValue(0);
		sliderMinFeret.setUpperValue(100);
		sliderMinFeret.setEnabled(true);
		GridBagConstraints gbc_sliderMinFeret = new GridBagConstraints();
		gbc_sliderMinFeret.fill = GridBagConstraints.BOTH;
		gbc_sliderMinFeret.insets = new Insets(0, 0, 5, 5);
		gbc_sliderMinFeret.gridx = 2;
		gbc_sliderMinFeret.gridy = 7;
		frmSemParticleSegmentation.getContentPane().add(sliderMinFeret, gbc_sliderMinFeret);
		
		JLabel lblNewLabel_2_2 = new JLabel("Min Feret Diameter:");
		GridBagConstraints gbc_lblNewLabel_2_2 = new GridBagConstraints();
		gbc_lblNewLabel_2_2.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2_2.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_2.gridx = 1;
		gbc_lblNewLabel_2_2.gridy = 7;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_2, gbc_lblNewLabel_2_2);
		
		JLabel lblNewLabel_2_2_1 = new JLabel("Area:");
		GridBagConstraints gbc_lblNewLabel_2_2_1 = new GridBagConstraints();
		gbc_lblNewLabel_2_2_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2_2_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_2_1.gridx = 1;
		gbc_lblNewLabel_2_2_1.gridy = 8;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_2_1, gbc_lblNewLabel_2_2_1);
		
		JLabel lblNewLabel_2_2_1_1 = new JLabel("Solidity:");
		GridBagConstraints gbc_lblNewLabel_2_2_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_2_2_1_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2_2_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_2_1_1.gridx = 1;
		gbc_lblNewLabel_2_2_1_1.gridy = 9;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_2_1_1, gbc_lblNewLabel_2_2_1_1);
		
		RangeSlider sliderSolidity = new RangeSlider();
		sliderSolidity.setValue(0);
		sliderSolidity.setUpperValue(100);
		sliderSolidity.setEnabled(true);
		GridBagConstraints gbc_sliderSolidity = new GridBagConstraints();
		gbc_sliderSolidity.fill = GridBagConstraints.BOTH;
		gbc_sliderSolidity.insets = new Insets(0, 0, 5, 5);
		gbc_sliderSolidity.gridx = 2;
		gbc_sliderSolidity.gridy = 9;
		frmSemParticleSegmentation.getContentPane().add(sliderSolidity, gbc_sliderSolidity);
		
		JLabel lblNewLabel_2_2_1_1_1 = new JLabel("Circularity:");
		GridBagConstraints gbc_lblNewLabel_2_2_1_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_2_2_1_1_1.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2_2_1_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_2_1_1_1.gridx = 1;
		gbc_lblNewLabel_2_2_1_1_1.gridy = 10;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_2_1_1_1, gbc_lblNewLabel_2_2_1_1_1);
		
		RangeSlider sliderCircularity = new RangeSlider();
		sliderCircularity.setValue(0);
		sliderCircularity.setUpperValue(100);
		sliderCircularity.setEnabled(true);
		GridBagConstraints gbc_sliderCircularity = new GridBagConstraints();
		gbc_sliderCircularity.fill = GridBagConstraints.BOTH;
		gbc_sliderCircularity.insets = new Insets(0, 0, 5, 5);
		gbc_sliderCircularity.gridx = 2;
		gbc_sliderCircularity.gridy = 10;
		frmSemParticleSegmentation.getContentPane().add(sliderCircularity, gbc_sliderCircularity);
		
		JLabel lblNewLabel_2_2_1_1_1_1 = new JLabel("Mean Intensity:");
		GridBagConstraints gbc_lblNewLabel_2_2_1_1_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_2_2_1_1_1_1.anchor = GridBagConstraints.WEST;
		gbc_lblNewLabel_2_2_1_1_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_2_2_1_1_1_1.gridx = 1;
		gbc_lblNewLabel_2_2_1_1_1_1.gridy = 11;
		frmSemParticleSegmentation.getContentPane().add(lblNewLabel_2_2_1_1_1_1, gbc_lblNewLabel_2_2_1_1_1_1);
		
		RangeSlider sliderMeanIntensity = new RangeSlider();
		sliderMeanIntensity.setValue(0);
		sliderMeanIntensity.setUpperValue(100);
		sliderMeanIntensity.setEnabled(true);
		GridBagConstraints gbc_sliderMeanIntensity = new GridBagConstraints();
		gbc_sliderMeanIntensity.fill = GridBagConstraints.BOTH;
		gbc_sliderMeanIntensity.insets = new Insets(0, 0, 5, 5);
		gbc_sliderMeanIntensity.gridx = 2;
		gbc_sliderMeanIntensity.gridy = 11;
		frmSemParticleSegmentation.getContentPane().add(sliderMeanIntensity, gbc_sliderMeanIntensity);
		
		JLabel lblAutoFilterThreshold = new JLabel("Auto-Filter Threshold:");
		GridBagConstraints gbc_lblAutoFilterThreshold = new GridBagConstraints();
		gbc_lblAutoFilterThreshold.fill = GridBagConstraints.BOTH;
		gbc_lblAutoFilterThreshold.insets = new Insets(0, 0, 5, 5);
		gbc_lblAutoFilterThreshold.gridx = 1;
		gbc_lblAutoFilterThreshold.gridy = 12;
		frmSemParticleSegmentation.getContentPane().add(lblAutoFilterThreshold, gbc_lblAutoFilterThreshold);

		JSlider sliderAutoFilterThreshold = new JSlider();
		sliderAutoFilterThreshold.setValue(50);
		sliderAutoFilterThreshold.setEnabled(false);
		GridBagConstraints gbc_sliderAutoFilterThreshold = new GridBagConstraints();
		gbc_sliderAutoFilterThreshold.fill = GridBagConstraints.BOTH;
		gbc_sliderAutoFilterThreshold.insets = new Insets(0, 0, 5, 5);
		gbc_sliderAutoFilterThreshold.gridx = 2;
		gbc_sliderAutoFilterThreshold.gridy = 12;
		frmSemParticleSegmentation.getContentPane().add(sliderAutoFilterThreshold, gbc_sliderAutoFilterThreshold);

		JCheckBox chckbxAutoFilterCheckBox = new JCheckBox("Use Auto-Filtering");
		chckbxAutoFilterCheckBox.setToolTipText("Use Neural-Network based classification for Filtering");
		chckbxAutoFilterCheckBox.setSelected(false);
		GridBagConstraints gbc_chckbxAutoFilterCheckBox = new GridBagConstraints();
		gbc_chckbxAutoFilterCheckBox.anchor = GridBagConstraints.WEST;
		gbc_chckbxAutoFilterCheckBox.fill = GridBagConstraints.VERTICAL;
		gbc_chckbxAutoFilterCheckBox.insets = new Insets(0, 0, 5, 5);
		gbc_chckbxAutoFilterCheckBox.gridx = 1;
		gbc_chckbxAutoFilterCheckBox.gridy = 13;
		frmSemParticleSegmentation.getContentPane().add(chckbxAutoFilterCheckBox, gbc_chckbxAutoFilterCheckBox);

		JButton btnNewButton = new JButton("Start");
		btnNewButton.setToolTipText("Start Image Segmentation");
		GridBagConstraints gbc_btnNewButton = new GridBagConstraints();
		gbc_btnNewButton.fill = GridBagConstraints.BOTH;
		gbc_btnNewButton.insets = new Insets(0, 0, 0, 5);
		gbc_btnNewButton.gridx = 1;
		gbc_btnNewButton.gridy = 14;
		frmSemParticleSegmentation.getContentPane().add(btnNewButton, gbc_btnNewButton);
		
		
		modelSelection = comboBox.getSelectedItem().toString();
		minThreshold = (float)sliderThreshold.getValue()/100.0f;
		minThresholdAutoFilter = (float)sliderAutoFilterThreshold.getValue()/100.0f;
		applyWatershed = chckbxNewCheckBox.isSelected();
		
		// Event Listeners:
		btnNewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				ImagePlus imp = null;
				if (WindowManager.getImageCount() == 0) {
					imp = IJ.openImage();
					if (imp == null) { // Open File Dialog was cancelled
						return;	
					}
					imp.show();
				} else {
					imp = IJ.getImage();
				}
				
				if (imp.getBitDepth() != 8 && imp.getBitDepth() != 16) {
					if (!IJ.showMessageWithCancel("SEM Particle Segmentation", "Image " + imp.getTitle() + " is not an 8 or 16-bit grayscale Image. Convert?")) {
						return;
					} else {
						// Convert to 8 bit grayscale
						new ImageConverter(imp).convertToGray8();
					}
				}

				inputImage = imp;
				imageWidth = imp.getProcessor().getWidth();
				imageHeight = imp.getProcessor().getHeight();
				new Thread((new Runnable() {
					public void run() {
						try {
							runInference();
							if (chckbxAutoFilterCheckBox.isSelected()) {
								runAutoFilter();
							}
							
							sliderMinFeret.setValue(0);
							sliderArea.setValue(0);
							sliderSolidity.setValue(0);
							sliderCircularity.setValue(0);
							sliderMeanIntensity.setValue(0);
							sliderMinFeret.setUpperValue(100);
							sliderArea.setUpperValue(100);
							sliderSolidity.setUpperValue(100);
							sliderCircularity.setUpperValue(100);
							sliderMeanIntensity.setUpperValue(100);

							sliderAutoFilterThreshold.setValue(50);
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				})).start();				
			}
		});
		
		comboBox.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				if (e.getStateChange() == e.SELECTED) { //ItemListener fires twice: once for deselecting the old entry and once for selecting the new entry
					logSetting("Model", comboBox.getSelectedItem().toString());
					modelSelection = comboBox.getSelectedItem().toString();
				}
			}
		});

		chckbxNewCheckBox.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				logSetting("Apply Watershed", String.valueOf(chckbxNewCheckBox.isSelected()));
				applyWatershed = chckbxNewCheckBox.isSelected();
				if (inputImage == null || inputImage.getWindow() == null) {
					return;
				}
				sliderMinFeret.setValue(0);
				sliderArea.setValue(0);
				sliderSolidity.setValue(0);
				sliderCircularity.setValue(0);
				sliderMeanIntensity.setValue(0);
				sliderMinFeret.setUpperValue(100);
				sliderArea.setUpperValue(100);
				sliderSolidity.setUpperValue(100);
				sliderCircularity.setUpperValue(100);
				sliderMeanIntensity.setUpperValue(100);
				
				sliderAutoFilterThreshold.setValue(50);
				minThreshold = (float)sliderThreshold.getValue()/100.0f;
				minThresholdAutoFilter = (float)sliderAutoFilterThreshold.getValue()/100.0f;
				segmented.close();
				segmented = segment();
				segmented.show();
				ij.WindowManager.toFront(ij.WindowManager.getWindow(inputImage.getTitle()));
				doAnalysis();
				applyFilterSettings("Circularity", 0.0f, 1.0f);
			}
		});

		chckbxAutoFilterCheckBox.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				logSetting("Auto Filter Enabled", String.valueOf(chckbxAutoFilterCheckBox.isSelected()));
				sliderMinFeret.setValue(0);
				sliderArea.setValue(0);
				sliderSolidity.setValue(0);
				sliderCircularity.setValue(0);
				sliderMeanIntensity.setValue(0);
				sliderMinFeret.setUpperValue(100);
				sliderArea.setUpperValue(100);
				sliderSolidity.setUpperValue(100);
				sliderCircularity.setUpperValue(100);
				sliderMeanIntensity.setUpperValue(100);

				sliderAutoFilterThreshold.setValue(50);
				sliderMinFeret.setEnabled(!chckbxAutoFilterCheckBox.isSelected());
				sliderArea.setEnabled(!chckbxAutoFilterCheckBox.isSelected());
				sliderSolidity.setEnabled(!chckbxAutoFilterCheckBox.isSelected());
				sliderCircularity.setEnabled(!chckbxAutoFilterCheckBox.isSelected());
				sliderMeanIntensity.setEnabled(!chckbxAutoFilterCheckBox.isSelected());
				sliderAutoFilterThreshold.setEnabled(chckbxAutoFilterCheckBox.isSelected());
				//minThreshold = (float)sliderThreshold.getValue()/100.0f;
				if (inputImage == null || predicted == null  || inputImage.getWindow() == null || predicted.getWindow() == null) {
					return;
				}
				if (chckbxAutoFilterCheckBox.isSelected()) {
					if (classified != null) {
						classified.close();
					}
					if (segmented != null) {
						segmented.close();
					}
					segmented = segment();
					segmented.show();
					new Thread((new Runnable() {
						public void run() {
							try {
								runAutoFilter();
							} catch (Exception e) {
								e.printStackTrace();
							}
						}
					})).start();				
				} else {
					doAnalysis();
					applyFilterSettings("Circularity", 0.0f, 1.0f);
				}
			}
		});
		
		sliderThreshold.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				JSlider source = (JSlider)e.getSource();
				if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
					minThreshold = (float)source.getValue()/100.0f;
					sliderMinFeret.setValue(0);
					sliderArea.setValue(0);
					sliderSolidity.setValue(0);
					sliderCircularity.setValue(0);
					sliderMeanIntensity.setValue(0);
					sliderMinFeret.setUpperValue(100);
					sliderArea.setUpperValue(100);
					sliderSolidity.setUpperValue(100);
					sliderCircularity.setUpperValue(100);
					sliderMeanIntensity.setUpperValue(100);
					segmented.close();
					segmented = segment();
					segmented.show();
					ij.WindowManager.toFront(ij.WindowManager.getWindow(inputImage.getTitle()));
					doAnalysis();
					applyFilterSettings("Circularity", 0.0f, 1.0f);
					logSetting("Threshold", String.valueOf(minThreshold));
				}
			}
		});
		
		sliderMinFeret.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	RangeSlider source = (RangeSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
			  		float minFeret = (float)source.getValue()/100.0f*minMaxAllFerets[1];
					float maxFeret = (float)source.getUpperValue()/100.0f*minMaxAllFerets[1];
  			  		applyFilterSettings("Min Feret", minFeret, maxFeret);
					logSetting("Ferets Diameter Range", String.valueOf(minFeret) + " - " + String.valueOf(maxFeret));
			  	}
			}
		});

		sliderArea.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	RangeSlider source = (RangeSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
			  		float minArea = (float)source.getValue()/100.0f*minMaxAllAreas[1];
					float maxArea = (float)source.getUpperValue()/100.0f*minMaxAllAreas[1];
			  		applyFilterSettings("Area", minArea, maxArea);
					logSetting("Area Range", String.valueOf(minArea) + " - " + String.valueOf(maxArea));
			  	}
			}
		});

		sliderSolidity.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	RangeSlider source = (RangeSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
			  		float minSolidity = (float)source.getValue()/100.0f*minMaxAllSolidities[1];
					float maxSolidity = (float)source.getUpperValue()/100.0f*minMaxAllSolidities[1];
			  		applyFilterSettings("Solidity", minSolidity, maxSolidity);
					logSetting("Solidity Range", String.valueOf(minSolidity) + " - " + String.valueOf(maxSolidity));
			  	}
			}
		});

		sliderCircularity.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	RangeSlider source = (RangeSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
			  		float minCircularity = (float)source.getValue()/100.0f*minMaxAllCircularities[1];
					float maxCircularity = (float)source.getUpperValue()/100.0f*minMaxAllCircularities[1];
			  		applyFilterSettings("Circularity", minCircularity, maxCircularity);
					logSetting("Circularity Range", String.valueOf(minCircularity) + " - " + String.valueOf(maxCircularity));
			  	}
			}
		});

		sliderMeanIntensity.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	RangeSlider source = (RangeSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || inputImage.getWindow() == null) {
						return;
					}
			  		float minMeanIntensity = (float)source.getValue()/100.0f*minMaxAllMeans[1];
					float maxMeanIntensity = (float)source.getUpperValue()/100.0f*minMaxAllMeans[1];
			  		applyFilterSettings("MeanIntensity", minMeanIntensity, maxMeanIntensity);
					logSetting("Mean Intensity Range", String.valueOf(minMeanIntensity) + " - " + String.valueOf(maxMeanIntensity));
			  	}
			}
		});
		
		sliderAutoFilterThreshold.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			  	JSlider source = (JSlider)e.getSource();
			  	if (!source.getValueIsAdjusting()) {
					if (inputImage == null || classified == null || inputImage.getWindow() == null) {
						return;
					}
			  		minThresholdAutoFilter = (float)source.getValue()/100.0f;
			  		
					roiMgr.runCommand(classified, "Measure");
					roiMgr.runCommand("Show None");
					resultsTableImage = ResultsTable.getResultsTable();
					allRois = roiMgr.getRoisAsArray().clone();
					if (allRois.length > 0) {
						allMeans = resultsTableImage.getColumn(resultsTableImage.getColumnIndex("Mean"));
						for (int i = 0; i < allMeans.length; i++) {
							allClasses[i] = allMeans[i];
						}
					}
					roiMgr.runCommand(inputImage, "Measure");
	
					applyFilterSettings("Classes", minThresholdAutoFilter, minThresholdAutoFilter);			  	
					logSetting("Auto-Filter Threshold", String.valueOf(minThresholdAutoFilter));
			  	}
			}
		});

		return frmSemParticleSegmentation;
	}

	/**
	 * Utility method to log changes in the filter options to the console window.
	 * 
	 * @param  prop {@code String} identifying the property that was changed
	 * @param  val {@code String} identifying the value the property was changed to
	 * @since 1.8
	 */
	private void logSetting(String prop, String val) {
		if (logTextWindow == null || logTextWindow.isShowing() == false) {
			logTextWindow = new TextWindow("SEM Particle Segmentation Log", "", 350, 300);
		}
		logTextWindow.setLocation(GUI.getLocation().x + GUI.getBounds().width, GUI.getLocation().y);
		logTextWindow.setSize(logTextWindow.getBounds().width, GUI.getBounds().height);
		ij.WindowManager.toFront(ij.WindowManager.getWindow(logTextWindow.getTitle()));
		logTextWindow.getTextPanel().append(prop + ": " + val);
		ij.WindowManager.toFront(ij.WindowManager.getWindow(logTextWindow.getTitle()));
	}
}
