import React from 'react'
import { NavLink} from 'react-router-dom'
// import '../styles/SymptomCard.css'
import '../styles/SymptomCardPills.css'

const SymptomLetterCard = ({ sympDict }) => {

    return (
        <div className='symptom-section'>
            {
                sympDict && Object.keys(sympDict).map((letter, index) => {
                    return (
                        <div key={index}>
                            <div className='card-wrapper'>
                                <h2>{letter}</h2>
                                <ul className='symptom-list'>
                                    {
                                        sympDict[letter].map((sym, index) => {
                                            return (
                                                <NavLink key={index} className="symptomName" to={sym}><p>{sym}</p></NavLink>
                                            )
                                        })
                                    }
                                </ul>
                            </div>
                            {/* {letter !== Object.keys(sympDict)[Object.keys(sympDict).length -1] ? <hr className='hr-line'/> : null} */}
                        </div>
                    
                    )
                })
            }
        </div>
    )
}

export default SymptomLetterCard