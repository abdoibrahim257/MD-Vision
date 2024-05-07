import React, {useEffect} from "react";
import NavBar from "./navbar";
import '../styles/heroSection.css';
import SearchBar from "./SearchBar.js";
import SearchBarList from "./SearchBarList.js";
import SymptomWelcomer from "./SymptomWelcomer.js";
import SymptomLetterCard from "./SymptomLetterCard.js";

const SymptomPage = () => {
    const [symptom, setSymptom] = React.useState('')
    const [results, setResults] = React.useState([])
    const [sympList, setSympList] = React.useState([]) //list of symptoms starting with the letter [A, B, C, ...
    

    /* first get symptom names from the backend  */
    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch("http://localhost:8000/maven");
            const data = await response.json();
            // console.log(data)
            const symptomsList = data.fileList;
            setSympList(symptomsList);
        }
        fetchData();
    }, [])


    // const sympNames = [ //examples
    //     "Abdominal pain",
    //     "Anxiety",
    //     "Back pain",
    //     "Bleeding",
    //     "Chest pain",
    //     "Cough",
    //     "Diarrhea",
    //     "Dizziness",
    //     "Fatigue",
    //     "Fever",
    //     "Headache",
    //     "Heartburn",
    //     "Joint pain",
    //     "Nausea",
    //     "Rash",
    //     "Shortness of breath",
    //     "Sore throat",
    //     "Vomiting"
    // ]


    const createSymDict = () => {
        let symDict = {};

        // Letters.forEach(letter => {
        //     symDict[letter] = []
        // })
        
        sympList.forEach(sym => {
            let firstLetter = sym[0].toUpperCase()
            if (!symDict[firstLetter]) {
                symDict[firstLetter] = []
            }
            symDict[firstLetter].push(sym)
        })

        return symDict;
    }
    // const dict = {}
    
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