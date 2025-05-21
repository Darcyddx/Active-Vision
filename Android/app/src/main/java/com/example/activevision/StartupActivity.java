package com.example.activevision;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import android.graphics.LinearGradient;
import android.graphics.Shader;
import android.graphics.Paint;
import android.widget.ImageButton;
import android.widget.TextView;

public class StartupActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_startup);


        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        TextView textView = findViewById(R.id.textView);
        textView.post(() -> {
            Paint paint = textView.getPaint();
            float width = paint.measureText(textView.getText().toString());

            // add gradient
            Shader shader = new LinearGradient(
                    0, 0, 0, textView.getHeight(),
                    new int[]{0xFF4CAF50, 0xFFFFEB3B, 0xFFFFFFFF},
                    null,
                    Shader.TileMode.CLAMP
            );

            textView.getPaint().setShader(shader);
            textView.invalidate();
        });



        ImageButton startButton = findViewById(R.id.btn_start);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(StartupActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }
}