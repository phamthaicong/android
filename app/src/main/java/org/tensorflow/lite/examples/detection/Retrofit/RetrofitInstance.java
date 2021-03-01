package org.tensorflow.lite.examples.detection.Retrofit;

import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitInstance {
    private static retrofit2.Retrofit retrofit;
//    private static final String BASE_LINK = "https://5f0fee4100d4ab001613442c.mockapi.io";
    private  static final  String BASE_LINK="https://checkin.nms.com.vn/nmsfaceid";



    public static retrofit2.Retrofit getRetrofitInstance() {
        if (retrofit == null) {
            retrofit = new retrofit2.Retrofit.Builder()
                    .baseUrl(BASE_LINK)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit;
    }
}
