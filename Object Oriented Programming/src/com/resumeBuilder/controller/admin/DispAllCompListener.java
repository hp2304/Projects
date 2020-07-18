package com.resumeBuilder.controller.admin;

import com.resumeBuilder.view.admin.mngCompany.AdmDispAllCompanies;
import com.resumeBuilder.view.admin.mngCompany.AdmMngCompanyView;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class DispAllCompListener implements ActionListener {

    private AdmDispAllCompanies dispAllCompsView;

    public DispAllCompListener(AdmDispAllCompanies dispAllCompsView) {
        this.dispAllCompsView = dispAllCompsView;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        try {
            AdmMngCompanyView frame = new AdmMngCompanyView();
            frame.setVisible(true);
            dispAllCompsView.setVisible(false);
        } catch (Exception error) {
            error.printStackTrace();
        }

    }


}
