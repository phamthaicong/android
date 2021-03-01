package org.tensorflow.lite.examples.detection.Retrofit;

import org.tensorflow.lite.examples.detection.Model.ChooseCheckIn;
import org.tensorflow.lite.examples.detection.Model.PostImage;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;

public interface Service {
    @POST("/user")
    Call<List<PostImage>> testRetrofit(@Body PostImage postImage);

    @POST("/check-in-tablet")
    Call<List<PostImage>> postImage(@Body PostImage postImage);
    
    @POST("/choose-to-check-in")
    Call<List<ChooseCheckIn>> chooseUserCheckIn(@Body  ChooseCheckIn chooseCheckIn);

    @GET("/user")
    Call<List<PostImage>> getUserSuccessTest();
}
