import axios from "axios";
import FormData from "form-data";

export default class ImageResizeService {
  buildImageResizeModel({ file, dimensions, crop }) {
    const seamCarvingFormData = new FormData();
    const { width, height } = dimensions;
    seamCarvingFormData.set("image", file);
    seamCarvingFormData.set(
      "crop_rows",
      parseInt(height) - parseInt(crop.height)
    );
    seamCarvingFormData.set(
      "crop_cols",
      parseInt(width) - parseInt(crop.width)
    );
    return seamCarvingFormData;
  }

  resizeImage(imageResizeModel) {
    return axios
      .post("http://localhost:5000/resize", imageResizeModel)
      .then((response) => {
        const { processed_images } = response.data;
        const [
          original_url,
          cropped_url,
          original_marked_url,
          energy_url,
          marked_energy_url,
        ] = processed_images;
        return {
          original_url,
          cropped_url,
          original_marked_url,
          energy_url,
          marked_energy_url,
        };
      });
  }
}
