import React from "react";
import NavBar from "./navbar";
import '../styles/heroSection.css';
import SearchBar from "./SearchBar.js";
import SearchBarList from "./SearchBarList.js";
import SymptomWelcomer from "./SymptomWelcomer.js";

const SymptomPage = () => {

    const [results, setResults] = React.useState([])

    return (
        <div>
            {/* an nave bar will be here */}
            <NavBar />

            <div className="content">
                <SymptomWelcomer />
                <SearchBar setResults  =  {setResults}/>
                <SearchBarList results = {results} />

            </div>
        </div>
    );
}

export default SymptomPage;