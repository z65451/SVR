import React from "react";
import ImageResultCard from "../ImageResultCard";

const ImageResizingResults = ({ resizeResultImages, resized }) => (
  <article
    className="tile is-ancestor"
    style={{ display: !resized ? "none" : "flex" }}
  >
    <ImageResultCard
      imageUrl={resizeResultImages && resizeResultImages.energy_url}
      imageName={"Energy computed from the image"}
    />
    <ImageResultCard
      imageUrl={resizeResultImages && resizeResultImages.marked_energy_url}
      imageName={"Optimal removal seams based on energy"}
    />
    <ImageResultCard
      imageUrl={resizeResultImages && resizeResultImages.original_marked_url}
      imageName={"Optimal removal seams on the image"}
    />
  </article>
);
export default ImageResizingResults;
