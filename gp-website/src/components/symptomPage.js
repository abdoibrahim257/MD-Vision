import React, {useEffect} from "react";
import NavBar from "./navbar";
import '../styles/heroSection.css';
import '../styles/SymptomPage.css';
import SearchBar from "./SearchBar.js";
import SearchBarList from "./SearchBarList.js";
import SymptomWelcomer from "./SymptomWelcomer.js";
import SymptomLetterCard from "./SymptomLetterCard.js";

const SymptomPage = () => {
    const [padded, setPadded] = React.useState(false)
    const [results, setResults] = React.useState([])
    const [sympList, setSympList] = React.useState([]) //list of symptoms starting with the letter [A, B, C, ...
    

    /* first get symptom names from the backend  */
    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch("https://shad-honest-anchovy.ngrok-free.app/maven", {
                    headers: new Headers({
                    "ngrok-skip-browser-warning": "69420",
                }),
            });
            const data = await response.json();
            // console.log(data)
            const symptomsList = data.fileList;
            setSympList(symptomsList);
        }
        fetchData();
    }, [])


    const createSymDict = () => {
        let symDict = {};
        
        sympList.forEach(sym => {
            let firstLetter = sym[0].toUpperCase()
            if (!symDict[firstLetter]) {
                symDict[firstLetter] = []
            }
            symDict[firstLetter].push(sym)
        })

        return symDict;
    }
    
    const dict = createSymDict()
    
    return (
        <div>
            {/* an nave bar will be here */}
            <div className="Symptom-background">
                <NavBar setPadding={setPadded}/>
                <div className="content">
                    <div className={padded ? "maintain-content" : null}>
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