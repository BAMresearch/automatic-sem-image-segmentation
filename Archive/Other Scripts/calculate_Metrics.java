import trainableSegmentation.metrics.*;
import ij.IJ;
import ij.*;
import ij.process.*;
import ij.gui.*;
import ij.plugin.*;
import ij.text.*;
 
import java.io.IOException;
import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class calculate_Metrics implements PlugIn {
	private final String ROOT_DIR = ".\\";
	private final String INPUT_DIR = ROOT_DIR + "Masks_Predicted";
	private final String GROUNDTRUTH_DIR = ROOT_DIR + "TiO2_Masks_Manual_4Connected";

	private final boolean calculatePixelError = true;
	private final boolean calculateWarpingError = true;
	private final boolean calculateRandError = true;
	private final boolean calculateRandError2 = true;
	private final boolean calculateVInfo = true;

	private final int IMAGE_WIDTH = 1024;
	private final int IMAGE_HEIGHT = 712;
	private final double RESCALING = 1.0d;

	@Override
	public void run(String arg) {
		double allPE = 0.0d;
		double allWE = 0.0d;
		double allNE = 0.0d;
		double allRE = 0.0d;
		double allRE2 = 0.0d;
		double allVI = 0.0d;
		String[] results;
		int k = 1;
		
		File dir = new File(INPUT_DIR);
		File[] inputDirectories = dir.listFiles();
		
		dir = new File(GROUNDTRUTH_DIR);
		File[] groundTruthImages = dir.listFiles((d, name) -> name.endsWith(".tif"));
		ImagePlus[] groundTruthImps = new ImagePlus[groundTruthImages.length];
		int i = 0;
		for (File f : groundTruthImages) {
			//IJ.log(String.valueOf(f.getPath()));
			groundTruthImps[i] = new ImagePlus(String.valueOf(f.getPath()));
			groundTruthImps[i].setRoi(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
			groundTruthImps[i] = groundTruthImps[i].crop();
			groundTruthImps[i].setTitle(f.getName());
			i++;
		}
		
		results = new String[inputDirectories.length + 1];
		results[0] = "Model Name;Minimum Pixel Error;Minimum Warping Error;No of errors (splits + mergers pixels);Minimum foreground-restricted Rand error;Minimum foreground-restricted Rand error after thinning;Minimum foreground-restricted information theoretic score after thinning";
		IJ.log("---");
		IJ.log("Evaluating segmentation...");
		for (File f : inputDirectories) {
			File[] inputImages = f.listFiles((d, name) -> name.endsWith(".tif"));
			ImagePlus[] inputImps = new ImagePlus[inputImages.length];
			i = 0;
			for (File f2 : inputImages) {
				inputImps[i] = new ImagePlus(String.valueOf(f2.getPath()));
				inputImps[i].setRoi(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
				inputImps[i] = inputImps[i].crop();
				inputImps[i].setTitle(f2.getName());
				if (inputImps[i].getBitDepth() == 8) {
					inputImps[i] = new ImagePlus(f2.getName(), new FloatProcessor(inputImps[i].getProcessor().getFloatArray()));					
					inputImps[i].getProcessor().multiply(1.0d/255.0d);					
				}
				if (RESCALING != 1.0d) {
					inputImps[i].getProcessor().setInterpolationMethod(ImageProcessor.BICUBIC);
					inputImps[i] = new ImagePlus(f2.getName(), inputImps[i].getProcessor().resize(IMAGE_WIDTH));
					if (inputImps[i].getStatistics().min < 0.0d) {
						inputImps[i].getProcessor().add(Math.abs(inputImps[i].getStatistics().min));
					}
					if (inputImps[i].getStatistics().max > 1.0d) {
						inputImps[i].getProcessor().multiply(1.0d/inputImps[i].getStatistics().max);
					}
				}
				i++;
			}

			allPE = 0.0d;
			allWE = 0.0d;
			allNE = 0.0d;
			allRE = 0.0d;
			allRE2 = 0.0d;
			allVI = 0.0d;
			for (int j=0; j<groundTruthImps.length; j++) {
				IJ.log("\nPixel Errors (" + String.valueOf((k-1)*groundTruthImps.length+j) + "/"  + String.valueOf(groundTruthImps.length*inputDirectories.length) +  ")");
				if (calculatePixelError) {
				    IJ.log("\nCalculating pixel error for Model: " + String.valueOf(f.getName()) + "; Input Image: " + inputImps[j].getTitle()  + "; Ground Truth Image: " + groundTruthImps[j].getTitle());
				    PixelError metricPE = new PixelError(groundTruthImps[j], inputImps[j]);
				    double maxFScorePE = metricPE.getPixelErrorMaximalFScore( 0.0, 1.0, 0.1 ); 
				    IJ.log("Minimum pixel error: " + String.valueOf((1.0 - maxFScorePE)));
				    allPE += (1.0 - maxFScorePE);
				}
		
				IJ.log("\nWarpingErrors (" + String.valueOf((k-1)*groundTruthImps.length+j) + "/"  + String.valueOf(groundTruthImps.length*inputDirectories.length) +  ")");
				if (calculateWarpingError) {
				    IJ.log("\nCalculating warping error by minimizing splits and mergers for Model: " + String.valueOf(f.getName()) + "; Input Image: " + inputImps[j].getTitle()  + "; Ground Truth Image: " + groundTruthImps[j].getTitle());
				    WarpingError metricWE = new WarpingError(groundTruthImps[j], inputImps[j]);
				    double minThreshold = inputImps[j].getStatistics().min;
				    double maxThreshold = inputImps[j].getStatistics().max-0.1d;
				    double warpingError = metricWE.getMinimumSplitsAndMergersErrorValue( Math.floor(10.0d*minThreshold)/10.0d, Math.floor(10.0d*maxThreshold)/10.0d, 0.1, false, 20 );
				    IJ.log("Minimum warping error: " + warpingError);
				    IJ.log("# errors (splits + mergers pixels) = " + Math.round(warpingError * inputImps[j].getWidth() * inputImps[j].getHeight() * inputImps[j].getImageStackSize()));
				    allWE += warpingError;
				    allNE += Math.round(warpingError * inputImps[j].getWidth() * inputImps[j].getHeight() * inputImps[j].getImageStackSize());
				}
				  
				IJ.log("\nRandErrors (" + String.valueOf((k-1)*groundTruthImps.length+j) + "/"  + String.valueOf(groundTruthImps.length*inputDirectories.length) +  ")");
				if (calculateRandError) {   
				    IJ.log("\nCalculating maximal F-score of the foreground-restricted Rand index for Model: " + String.valueOf(f.getName()) + "; Input Image: " + inputImps[j].getTitle()  + "; Ground Truth Image: " + groundTruthImps[j].getTitle());
				    RandError metricRE = new RandError(groundTruthImps[j], inputImps[j]);
				    double maxFScoreRE = metricRE.getForegroundRestrictedRandIndexMaximalFScore( 0.0, 1.0, 0.1 );  
				    IJ.log("Minimum foreground-restricted Rand error: " + (1.0 - maxFScoreRE) );     
				    allRE += (1.0 - maxFScoreRE);
				} 

				IJ.log("\nRandErrorsAfterThinning (" + String.valueOf((k-1)*groundTruthImps.length+j) + "/"  + String.valueOf(groundTruthImps.length*inputDirectories.length) +  ")");
				if (calculateRandError2) {   
				    IJ.log("\nCalculating maximal F-score of the foreground-restricted Rand index after thinning for Model: " + String.valueOf(f.getName()) + "; Input Image: " + inputImps[j].getTitle()  + "; Ground Truth Image: " + groundTruthImps[j].getTitle());
				    RandError metricRE2 = new RandError(groundTruthImps[j], inputImps[j]);
				    double maxFScoreRE2 = metricRE2.getMaximalVRandAfterThinning( 0.0, 1.0, 0.1, true );  
				    IJ.log("Minimum Rand error after thinning: " + (1.0 - maxFScoreRE2) );     
				    allRE2 += (1.0 - maxFScoreRE2);
				} 

				IJ.log("\nInformationTheoreticScoreAfterThinning (" + String.valueOf((k-1)*groundTruthImps.length+j) + "/"  + String.valueOf(groundTruthImps.length*inputDirectories.length) +  ")");
				if (calculateVInfo) {   
				    IJ.log("\nCalculatingCalculating maximal foreground-restricted information theoretic score after thinning index for Model: " + String.valueOf(f.getName()) + "; Input Image: " + inputImps[j].getTitle()  + "; Ground Truth Image: " + groundTruthImps[j].getTitle());
				    VariationOfInformation metricVI = new VariationOfInformation(groundTruthImps[j], inputImps[j]);
				    double maxFScoreVI = metricVI.getMaximalVInfoAfterThinning( 0.0, 1.0, 0.1 );  
				    IJ.log("Minimum information theoretic score after thinning: " + (1.0 - maxFScoreVI) );     
				    allVI += (1.0 - maxFScoreVI);
				} 
			}
			results[k] = String.valueOf(f.getName()) + ";" + String.valueOf(allPE/groundTruthImps.length) + ";" + String.valueOf(allWE/groundTruthImps.length) + ";" + String.valueOf(allNE/groundTruthImps.length) + ";" + String.valueOf(allRE/groundTruthImps.length) + ";" + String.valueOf(allRE2/groundTruthImps.length) + ";" + String.valueOf(allVI/groundTruthImps.length); 
			k++;
		}
		
		String data = "";
	    IJ.log("\n\n");
		for (int j=0; j<results.length; j++) {
		    IJ.log(results[j] + "\n");
			data += results[j] + "\n";
		}
		
		try {
			Files.write(Paths.get(ROOT_DIR + "\\Metrics_Extended2.csv"), data.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }		
	}
}	  

