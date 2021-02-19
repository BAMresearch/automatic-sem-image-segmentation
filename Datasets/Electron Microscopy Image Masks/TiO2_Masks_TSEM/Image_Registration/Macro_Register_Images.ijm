inputPath = "C:/Users/bruehle/Documents/CNN/Dan/TiO2-NPs/";
outputPath = "C:/Users/bruehle/Desktop/Image_Registration/";
filenameConst=1908248;

for (i=filenameConst+42; i<filenameConst+80; i+=2) {
	open(inputPath+"SEM/"+toString(i)+".tif");
	curPath = outputPath + "Input/" + replace(getTitle(),".tif","");
	File.makeDirectory(curPath);
	File.makeDirectory(replace(curPath,"Input","Output"));
	run("8-bit");
	makeRectangle(0, 0, 1024, 712);
	run("Crop");
	saveAs("Tiff", curPath + "/" + getTitle());
	close();
	open(inputPath+"TSEM/"+toString(i+1)+".tif");
	run("8-bit");
	makeRectangle(0, 0, 1024, 712);
	run("Crop");
	run("Invert");
	saveAs("Tiff", curPath + "/" + getTitle());
	close();
	run("Register Virtual Stack Slices", "source=" + curPath + " output=" + replace(curPath,"Input","Output") + " feature=Translation registration=[Translation          -- no deformation                      ] save");
	close();
}