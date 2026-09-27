import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageButton;

import com.google.android.material.slider.RangeSlider;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {
    private DrawView paint;
    private ImageButton save,undo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        paint=(DrawView)findViewById(R.id.draw_view);
        undo=(ImageButton)findViewById(R.id.btn_undo);
        save=(ImageButton)findViewById(R.id.btn_save);

        Double[] biases = {2.15538e-019,-1.83537e-014,1.32787e-009,0.000121595,0.692,-9.69657e-022,-3.91183e-017,0.322554,-1.46034e-013,-2.17969e-008,3.78937e-007,-1.04109,-8.36949e-012,6.25534e-023,2.82682e-006,-0.0502846,1.11272e-012,-1.63543e-011,-5.11531e-006,2.23902e-013,-4.14867e-007,1.18977e-008,0.697538,-0.00046909,-7.75604e-014,2.33036e-005,-1.81155e-014,1.98464e-009,5.43193e-010,1.68703e-008,2.38575e-029,-0.000611377,-0.322114,-0.00749812,-1.1518,-0.0401759,-0.00700459,0.000681831,-2.35221,-0.52331,-0.000205428,-0.0274024};
        Double[] heuristics = new Double[28*28*32 + 10*32];

        BufferedReader reader;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open("netwerkWaardes.txt")));
            String mLine = reader.readLine();
            String[] strHeuristics = mLine.split(",");
            Log.d("aaa", Integer.toString(strHeuristics.length));
            for (int i = 0; i < strHeuristics.length; i++) {
                heuristics[i] = Double.parseDouble(strHeuristics[i]);
            }
        } catch (IOException e) {
            Log.d("main","file niet gevonden");
        }


        NeuralNet neuralNet = new NeuralNet(28 * 28, 1, 32, 2, biases, heuristics);

        undo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                paint.undo();
            }
        });

        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                double[] inputArr = new double[28 * 28];
                Bitmap bmp=paint.save();
                int bmpHeight = bmp.getHeight();
                int bmpWidth = bmp.getWidth();
                int blockSize = bmpHeight/28;
                int[][] raw = new int[bmpHeight][bmpHeight];
                for (int x = 0; x < bmpHeight; x++) {
                    //String line = "";
                    for (int y = 0; y < bmpHeight; y++) {
                        int relX = x - ((bmpHeight/2) - (bmpWidth/2));
                        if (relX < 0 || relX >= bmpWidth)
                            raw[x][y] = 0;
                        else {
                            int col = 255 - Color.red(bmp.getPixel(relX, y));
                            raw[x][y] = col;
                        }
                        /*if (raw[x][y] == 255)
                            line += "X";
                        else
                            line += "-";*/
                    }
                    //Log.v(Integer.toString(x), line);
                }

                int[][] step1 = new int[28][bmpHeight];

                for (int y = 0; y < bmpHeight; y++) {
                    for (int nx = 0; nx < 28; nx++) {
                        int inp = 0;
                        for (int x = nx * blockSize; x < (nx + 1) * blockSize; x++) {
                            inp += raw[x][y];
                        }
                        step1[nx][y] = inp / blockSize;
                    }
                }

                int[][] step2 = new int[28][28];

                for (int x = 0; x < 28; x++) {
                    for (int ny = 0; ny < 28; ny++) {
                        int inp = 0;
                        for (int y = ny * blockSize; y < (ny + 1) * blockSize; y++) {
                            inp += step1[x][y];
                        }
                        step2[x][ny] = inp / blockSize;
                    }
                }

                int[][] step3 = new int [28][28];

                int avg = 0;
                int sum = 0;
                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        sum += step2[x][y];
                        avg += step2[x][y] * x;
                        step3[x][y] = 0;
                    }
                }

                if (sum != 0) avg /= sum;
                else avg = 15;

                if (avg < 15)  {
                    int dif = 15 - avg;
                    for (int x = dif; x < 28; x++) {
                        for (int y = 0; y < 28; y++) {
                            step3[x][y] = step2[x - dif][y];
                        }
                    }
                }
                else if (avg > 15) {
                    int dif = avg - 15;
                    for (int x = dif; x < 28; x++) {
                        for (int y = 0; y < 28; y++) {
                            step3[x - dif][y] = step2[x][y];
                        }
                    }
                }

                int[][] step4 = new int [28][28];

                avg = 0;
                sum = 0;
                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        sum += step2[x][y];
                        avg += step2[x][y] * y;
                        step4[x][y] = 0;
                    }
                }

                if (sum != 0) avg /= sum;
                else avg = 15;

                if (avg < 15)  {
                    int dif = 15 - avg;
                    for (int x = 0; x < 28; x++) {
                        for (int y = dif; y < 28; y++) {
                            step4[x][y] = step3[x][y - dif];
                        }
                    }
                }
                else if (avg > 15) {
                    int dif = avg - 15;
                    for (int x = 0; x < 28; x++) {
                        for (int y = dif; y < 28; y++) {
                            step4[x][y - dif] = step3[x][y];
                        }
                    }
                }

                double[][] step5 = new double[28][28];

                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        step5[x][y] = ((1.0 * step4[x][y])/255.0) * 2.0 - 1.0;
                    }
                }

                for (int y = 0; y < 28; y++) {
                    String line = "";
                    for (int x = 0; x < 28; x++) {
                        if (step5[x][y] >= 0.8)
                            line += "X";
                        else if (step5[x][y] >= 0.2)
                            line += "+";
                        else if (step5[x][y] >= -0.3)
                            line += "-";
                        else if (step5[x][y] >= -0.7)
                            line += "'";
                        else
                            line += " ";
                    }
                    Log.d(Integer.toString(y),line);
                }

                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        inputArr[x + y * 28] = 1.0 * step5[x][y];
                    }
                }

                int answer = neuralNet.output(inputArr);

                AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
                alertDialog.setTitle("Hoi wouter <3");
                alertDialog.setMessage(Integer.toString(answer));
                alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int which) {
                                dialog.dismiss();
                            }
                        });
                alertDialog.show();
            }
        });

        ViewTreeObserver vto = paint.getViewTreeObserver();
        vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {

                paint.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int width = paint.getMeasuredWidth();
                int height = paint.getMeasuredHeight();
                paint.init(height, width);
            }
        });
    }
}