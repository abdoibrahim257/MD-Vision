import React from "react";
import NavBar from "./navbar";
import '../styles/heroSection.css';
import SearchBar from "./SearchBar.js";
import SearchBarList from "./SearchBarList.js";
import SymptomWelcomer from "./SymptomWelcomer.js";
import SymptomLetterCard from "./SymptomLetterCard.js";

const SymptomPage = () => {
    const [symptom, setSymptom] = React.useState('')
    const [results, setResults] = React.useState([])
    
    /* first get symptom names from the backend  */
    const sympNames = [ //examples
        "Abdominal pain",
        "Anxiety",
        "Back pain",
        "Bleeding",
        "Chest pain",
        "Cough",
        "Diarrhea",
        "Dizziness",
        "Fatigue",
        "Fever",
        "Headache",
        "Heartburn",
        "Joint pain",
        "Nausea",
        "Rash",
        "Shortness of breath",
        "Sore throat",
        "Vomiting"
    ]

    // const Letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    const createSymDict = () => {
        let symDict = {};

        // Letters.forEach(letter => {
        //     symDict[letter] = []
        // })
            
        sympNames.forEach(sym => {
            let firstLetter = sym[0].toUpperCase()
            if (!symDict[firstLetter]) {
                symDict[firstLetter] = []
            }
            symDict[firstLetter].push(sym)
        })

        return symDict;
    }
    
    const dict = createSymDict()

    // console.log(symDict)
    
    return (
        <div>
            {/* an nave bar will be here */}
            <div className="background">
                <NavBar sticky />
                <div className="content">
                    <div>
                        <SymptomWelcomer />
                        <SearchBar setResults  =  {setResults}/>
                        <SearchBarList results = {results} />
                    </div>
                </div>
            </div>
            <div className="content sypmtom-list">
                {/* split them into dict where key is the first letter and value is the list of symptoms starting with that letter */}
                <SymptomLetterCard sympDict = {dict} />
            </div>
        </div>
    );
}

export default SymptomPage;