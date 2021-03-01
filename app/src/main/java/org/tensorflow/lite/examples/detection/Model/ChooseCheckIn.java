package org.tensorflow.lite.examples.detection.Model;

public class ChooseCheckIn {
    String emp_id,image;

    public ChooseCheckIn() {
    }

    public ChooseCheckIn(String emp_id, String image) {
        this.emp_id = emp_id;
        this.image = image;
    }

    public String getEmp_id() {
        return emp_id;
    }

    public void setEmp_id(String emp_id) {
        this.emp_id = emp_id;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }
}
