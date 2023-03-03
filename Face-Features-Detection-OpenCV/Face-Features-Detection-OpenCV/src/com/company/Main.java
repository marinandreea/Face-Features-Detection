package com.company;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Main {

    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    public static void main(String[] args) {

        CascadeClassifier faceCascade = new CascadeClassifier();
        CascadeClassifier eyeCascade = new CascadeClassifier();
        CascadeClassifier noseCascade = new CascadeClassifier();
        CascadeClassifier mouthCascade = new CascadeClassifier();

        if(!faceCascade.load("C:\\Users\\Andreea\\OneDrive\\Desktop\\AN 3\\SCS\\Face-Features-Detection-OpenCV\\Face-Features-Detection-OpenCV\\resource\\haarcascade_frontalface_alt.xml")){
            System.out.println("could not load haarcascade_frontalface_alt.xml");
            return;
        }
        if(!eyeCascade.load("C:\\Users\\Andreea\\OneDrive\\Desktop\\AN 3\\SCS\\Face-Features-Detection-OpenCV\\Face-Features-Detection-OpenCV\\resource\\haarcascade_eye_tree_eyeglasses.xml")){
            System.out.println("could not load haarcascade_eye_tree_eyeglasses.xml");
            return;
        }

        if(!noseCascade.load("C:\\Users\\Andreea\\OneDrive\\Desktop\\AN 3\\SCS\\Face-Features-Detection-OpenCV\\Face-Features-Detection-OpenCV\\resource\\nose.xml")){
            System.out.println("could not load nose.xml");
            return;
        }
        if(!mouthCascade.load("C:\\Users\\Andreea\\OneDrive\\Desktop\\AN 3\\SCS\\Face-Features-Detection-OpenCV\\Face-Features-Detection-OpenCV\\resource\\mouth.xml")){
            System.out.println("could not load mouth.xml");
            return;
        }


        VideoCapture cap = new VideoCapture(0);
        Mat frame = new Mat();


        Scalar col1 = new Scalar(150, 70, 240);
        Scalar col2 = new Scalar(200, 300, 255);
        Scalar col3 = new Scalar(500, 214, 100);
        Scalar col4 = new Scalar(100, 459, 79);

        if(!cap.isOpened()){
            System.out.println("Camera not found! Please try again!");
            return;
        }

        while (true) {
            cap.read(frame);

            Mat gray = new Mat();

            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(gray, gray);
            MatOfRect faces = new MatOfRect();
            faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, new Size(100, 100));

            Rect[] faceRects = faces.toArray();
            if(faceRects.length>0) {
                for (Rect r : faceRects) {
                    Imgproc.rectangle(frame, r, col1, 2);
                    int font = Imgproc.FONT_HERSHEY_DUPLEX;

                    Point startingPoint = new Point(r.tl().x-1,r.br().y);
                    Point endingPoint = new Point(r.br().x,r.br().y+25);
                    Imgproc.rectangle(frame,startingPoint,endingPoint,col1,-1);

                    double fontScale = (r.br().x/r.tl().x)/3;

                    Imgproc.putText(frame,"Person Detected",new Point(r.tl().x+5,r.br().y+18),font,fontScale,new Scalar(255, 255, 255),1);

                    Mat face = new Mat(frame,r);

                    //eyes
                    MatOfRect eyes = new MatOfRect();
                    eyeCascade.detectMultiScale(face,eyes,1.1,3,0);
                    Rect[] eyeRects = eyes.toArray();

                    if(eyeRects.length<3 && eyeRects.length>0)
                    {
                        for (Rect e : eyeRects) {
                            Point tl = new Point(r.tl().x + e.tl().x, r.tl().y + e.tl().y);
                            Point br = new Point(r.tl().x + e.br().x, r.tl().y + e.br().y);

                            Imgproc.rectangle(frame, tl, br, col2, 2);

                            Imgproc.line(frame, br, new Point(frame.width() - 145, br.y), col2, 2);

                            Imgproc.putText(frame, "Eyes Detected", new Point(frame.width() - 140, br.y), font, 0.5, new Scalar(255, 255, 255), 1);

                        }
                    }

                    //nose
                    MatOfRect nose = new MatOfRect();
                    noseCascade.detectMultiScale(face,nose,1.1,5,0);
                    Rect[] noseRects = nose.toArray();

                    if(noseRects.length==1) {
                        for (Rect n : noseRects) {
                            Point tl = new Point(r.tl().x + n.tl().x, r.tl().y + n.tl().y);
                            Point br = new Point(r.tl().x + n.br().x, r.tl().y + n.br().y);

                            Imgproc.rectangle(frame, tl, br, col3, 2);

                            Imgproc.line(frame, br, new Point(frame.width() - 205, br.y), col3, 2);

                            Imgproc.putText(frame, "Nose Detected", new Point(frame.width() - 200, br.y), font, 0.5, new Scalar(255, 255, 255), 1);

                        }
                    }

                    //mouth
                    MatOfRect mouth = new MatOfRect();
                    mouthCascade.detectMultiScale(face,mouth,1.5,5,0);
                    Rect[] mouthRects = mouth.toArray();

                    if(mouthRects.length==1) {
                        for (Rect m : mouthRects) {
                            Point tl = new Point(r.tl().x + m.tl().x, r.tl().y + m.tl().y);
                            Point br = new Point(r.tl().x + m.br().x, r.tl().y + m.br().y);

                            Imgproc.rectangle(frame, tl, br, col4, 2);

                            Imgproc.line(frame, br, new Point(frame.width() - 155, br.y), col4, 2);

                            Imgproc.putText(frame, "Mouth Detected", new Point(frame.width() - 150, br.y), font, 0.5, new Scalar(255, 255, 255), 1);

                        }
                    }

                }
            }
            HighGui.imshow("Feature Detection",frame);
            if (HighGui.waitKey(10)>=0) break;
        }
    }
}
