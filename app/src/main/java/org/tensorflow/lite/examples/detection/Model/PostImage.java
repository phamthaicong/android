package org.tensorflow.lite.examples.detection.Model;

public class PostImage {
    String image,platform;


    public PostImage() {
    }

    public PostImage(String image, String platform) {
        this.image = image;
        this.platform = platform;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public String getPlatform() {
        return platform;
    }

    public void setPlatform(String platform) {
        this.platform = platform;
    }
}
