package training.data.loader;

/**
 * Created by inakov on 5/4/15.
 */

import org.apache.commons.math3.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * This class implements a reader for the MNIST dataset of handwritten digits. The dataset is found
 * at http://yann.lecun.com/exdb/mnist/.
 *
 * @author Gabe Johnson <johnsogg@cmu.edu>
 */
public class MNISTDataUtil {

    public static void main(String[] args) throws IOException {

        final String imagesFilePath = "/home/inakov/IdeaProjects/branches/branches/branches/nn-deeplearning/src/main/resources/train-images-idx3-ubyte";
        final String labelsFilePath = "/home/inakov/IdeaProjects/branches/branches/branches/nn-deeplearning/src/main/resources/train-labels-idx1-ubyte";

        Pair<List<int[][]>,List<Integer>> trainingData = loadTrainingData(imagesFilePath, labelsFilePath);

        List<int[][]> imagesList = trainingData.getFirst();
        List<Integer> labelsList = trainingData.getSecond();

        displayImage(imagesList.get(10), labelsList.get(10));

    }

    public static Pair<List<int[][]>,List<Integer>> loadTrainingData(String imagesFilePath, String labelsFilePath) throws FileNotFoundException {
        Pair<List<int[][]>,List<Integer>> result = null;

        try {
            List<int[][]> imagesList = new LinkedList<int[][]>();
            List<Integer> labelsList = new LinkedList<Integer>();

            DataInputStream labels = new DataInputStream(new FileInputStream(labelsFilePath));
            DataInputStream images = new DataInputStream(new FileInputStream(imagesFilePath));
            int magicNumber = labels.readInt();
            if (magicNumber != 2049) {
                System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
                System.exit(0);
            }
            magicNumber = images.readInt();
            if (magicNumber != 2051) {
                System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
                System.exit(0);
            }
            int numLabels = labels.readInt();
            int numImages = images.readInt();
            int numRows = images.readInt();
            int numCols = images.readInt();
            if (numLabels != numImages) {
                System.err.println("Image file and label file do not contain the same number of entries.");
                System.err.println("  Label file contains: " + numLabels);
                System.err.println("  Image file contains: " + numImages);
                System.exit(0);
            }

            long start = System.currentTimeMillis();
            int numLabelsRead = 0;
            int numImagesRead = 0;
            while (labels.available() > 0 && numLabelsRead < numLabels) {
                byte label = labels.readByte();
                labelsList.add((int) label);
                numLabelsRead++;
                int[][] image = new int[numCols][numRows];
                for (int colIdx = 0; colIdx < numCols; colIdx++) {
                    for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                        image[colIdx][rowIdx] = images.readUnsignedByte();
                    }
                }
                numImagesRead++;
                // At this point, 'label' and 'image' agree and you can do whatever you like with them.
                imagesList.add(image);

                if (numLabelsRead % 10 == 0) {
                    System.out.print(".");
                }
                if ((numLabelsRead % 800) == 0) {
                    System.out.print(" " + numLabelsRead + " / " + numLabels);
                    long end = System.currentTimeMillis();
                    long elapsed = end - start;
                    long minutes = elapsed / (1000 * 60);
                    long seconds = (elapsed / 1000) - (minutes * 60);
                    System.out.println("  " + minutes + " m " + seconds + " s ");
                }
            }
            System.out.println();
            long end = System.currentTimeMillis();
            long elapsed = end - start;
            long minutes = elapsed / (1000 * 60);
            long seconds = (elapsed / 1000) - (minutes * 60);
            System.out
                    .println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");

            System.out.println("ImagesList size:" + imagesList.size());
            System.out.println("Image: " + imagesList.get(0).length + "x" + imagesList.get(0)[0].length);

            result = new Pair<List<int[][]>, List<Integer>>(imagesList, labelsList);
        }catch (IOException e){
            System.out.println("File not found!");
        }

        return result;
    }

    public static void displayImage(int[][] data, int label){
        System.out.println("Label: " + label);
        final int WIDTH = 28;
        final int HEIGHT = 28;

        final BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D)img.getGraphics();
        for(int i = 0; i < WIDTH; i++) {
            for(int j = 0; j < HEIGHT; j++) {
                int c =  data[j][i];
                g.setColor(new Color(c, c, c));
                g.fillRect(i, j, 1, 1);
            }
        }

        JFrame frame = new JFrame("Image test");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2d = (Graphics2D)g;
                g2d.clearRect(0, 0, getWidth(), getHeight());
                g2d.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                // Or _BICUBIC
                g2d.scale(2, 2);
                g2d.drawImage(img, 0, 0, this);
            }
        };
        panel.setPreferredSize(new Dimension(WIDTH*2, HEIGHT*2));
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

}