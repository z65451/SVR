import React from "react";
import FullscreenImageModel from "./FullscreenImageModal";

const ImageResultCard = ({ imageUrl, imageName }) => {
  const actualImageUrl = imageUrl ? imageUrl : "";

  return (
    <div className="tile is-parent has-text-centered">
      <article className="tile is-child box">
        <p className="subtitle">{imageName}</p>
        <div className="is-flex is-horizontal-center">
          <div>
            <div className="field is-fullwidth">
              <figure className="image">
                <img src={actualImageUrl}></img>
              </figure>
            </div>
            <FullscreenImageModel imageUrl={actualImageUrl} />
          </div>
        </div>
      </article>
    </div>
  );
};

export default ImageResultCard;
