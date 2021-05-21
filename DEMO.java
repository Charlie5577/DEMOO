

public class TensorFlowObjectDetectionAPIModel implements Classifier {
private static final Logger LOGGER = new Logger();
private static final int MAX_RESULTS = 100;
private String inputName;
private int inputSize;
private Vector<String> labels = new Vector<String>();
private int[] intValues;
private byte[] byteValues;
private float[] outputLocations;
private float[] outputScores;
private float[] outputClasses;
private float[] outputNumDetections;
private String[] outputNames;

private boolean logStats = false;

private TensorFlowInferenceInterface inferenceInterface;
public static Classifier create(
final AssetManager assetManager,
final String modelFilename,
final String labelFilename,
final int inputSize) throws IOException {
final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();

InputStream labelsInput = null;
String actualFilename = labelFilename.split("file:///android_asset/")[1];
labelsInput = assetManager.open(actualFilename);
BufferedReader br = null;
br = new BufferedReader(new InputStreamReader(labelsInput));
String line;
while ((line = br.readLine()) != null) {
LOGGER.w(line);
d.labels.add(line);
}
br.close();

d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

final Graph g = d.inferenceInterface.graph();

d.inputName = "image_tensor";
final Operation inputOp = g.operation(d.inputName);
if (inputOp == null) {
throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
}
d.inputSize = inputSize;

final Operation outputOp1 = g.operation("detection_scores");
if (outputOp1 == null) {
throw new RuntimeException("Failed to find output Node 'detection_scores'");
}
final Operation outputOp2 = g.operation("detection_boxes");
if (outputOp2 == null) {
throw new RuntimeException("Failed to find output Node 'detection_boxes'");
}
final Operation outputOp3 = g.operation("detection_classes");
if (outputOp3 == null) {
throw new RuntimeException("Failed to find output Node 'detection_classes'");
}

d.outputNames = new String[] {"detection_boxes", "detection_scores",
"detection_classes", "num_detections"};
d.intValues = new int[d.inputSize * d.inputSize];
d.byteValues = new byte[d.inputSize * d.inputSize * 3];
d.outputScores = new float[MAX_RESULTS];
d.outputLocations = new float[MAX_RESULTS * 4];
d.outputClasses = new float[MAX_RESULTS];
d.outputNumDetections = new float[1];
return d;
}      
