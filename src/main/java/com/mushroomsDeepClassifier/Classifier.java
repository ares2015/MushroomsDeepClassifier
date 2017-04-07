package com.mushroomsDeepClassifier;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Random;

/**
 * Created by Oliver on 4/7/2017.
 */
public class Classifier {

    protected static final Logger log = LoggerFactory.getLogger(Trainer.class);
    protected static int height = 400;
    protected static int width = 400;
    protected static int channels = 3;
    protected static int numExamples = 30;
    protected static int numLabels = 4;
    protected static int batchSize = 10;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 7;
    protected static double splitTrainTest = 0.8;
    protected static int nCores = 2;
    protected static boolean save = true;

    protected static String modelType = "AlexNet"; //

    public static void main(String[] args) throws Exception {

        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(basePath + "model.bin", true);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/mushrooms/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);


        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 1);
        InputSplit testData = inputSplit[0];


        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(testData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        while (dataIter.hasNext()) {
            DataSet testDataSet = dataIter.next();

            String expectedResult = testDataSet.getLabelName(0);
            List<String> predict = multiLayerNetwork.predict(testDataSet);
            String modelResult = predict.get(0);
            System.out.print("\nFor example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");
        }
    }


}
