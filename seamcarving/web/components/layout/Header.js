import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faImage } from "@fortawesome/free-solid-svg-icons";

const Header = () => (
  <section className="hero is-primary">
    <div className="hero-body">
      <div className="container">
        <h1 className="title">
          <FontAwesomeIcon icon={faImage} />
          &nbsp; Image seam carving
        </h1>
        <h2 className="subtitle">context-aware resizing demo app</h2>
      </div>
    </div>
  </section>
);

export default Header;
